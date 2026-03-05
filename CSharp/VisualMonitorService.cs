using OpenCvSharp;
using OpenCvSharp.Extensions;   // BitmapConverter (OpenCvSharp4.Extensions NuGet)
using System.Drawing;
using System.Net.Http.Headers;
using System.Text.Json;
using TestFuturisticAnomalyDetection.Models;

namespace TestFuturisticAnomalyDetection.Services
{
    /// <summary>
    /// Manages the Arducam OV9281 camera:
    ///   • Runs a persistent 15-fps preview loop that fires <see cref="FrameReady"/>.
    ///   • Captures JPEG frames and MP4 clips for the visual-inspection REST endpoints.
    ///   • All methods are fail-safe: exceptions are logged, not re-thrown.
    /// </summary>
    public class VisualMonitorService : IDisposable
    {
        // ── Constants ────────────────────────────────────────────────────────────
        private const string ImagePath         = "check-visual-image/";
        private const string VideoPath         = "check-visual-video/";
        private const int    PreviewFps        = 15;
        private const int    ClipDurationSecs  = 2;
        private const int    ClipFps           = 30;

        // ── Fields ───────────────────────────────────────────────────────────────
        private readonly ILogger    _logger;
        private readonly HttpClient _http;
        private readonly string     _imageEndpoint;
        private readonly string     _videoEndpoint;
        private readonly int        _cameraIndex;

        // Shared capture (kept open for the duration of machine run)
        private VideoCapture?         _capture;
        private Thread?               _previewThread;
        private volatile bool         _previewRunning;
        private readonly object       _captureLock = new();

        // Latest raw frame for anomaly captures (avoids reopening the camera)
        private Mat?                  _latestFrame;
        private readonly object       _frameLock = new();

        private bool _disposed;

        // ── Events ───────────────────────────────────────────────────────────────
        /// <summary>Raised on the thread-pool at ~15 fps with a fresh Bitmap clone. Caller must Dispose.</summary>
        public event EventHandler<Bitmap>? FrameReady;

        // ── Constructor ──────────────────────────────────────────────────────────
        public VisualMonitorService(ILogger logger, string visualApiBaseUrl, int cameraIndex = 1)
        {
            _logger      = logger;
            _cameraIndex = cameraIndex;
            _http        = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };

            if (!visualApiBaseUrl.EndsWith("/"))
                visualApiBaseUrl += "/";

            _imageEndpoint = visualApiBaseUrl + ImagePath;
            _videoEndpoint = visualApiBaseUrl + VideoPath;

            _logger.WriteLog($"[VisualMonitor] Camera index  : {_cameraIndex}");
            _logger.WriteLog($"[VisualMonitor] Image endpoint: {_imageEndpoint}");
            _logger.WriteLog($"[VisualMonitor] Video endpoint: {_videoEndpoint}");
        }

        // ══════════════════════════════════════════════════════════════════════════
        // Preview lifecycle
        // ══════════════════════════════════════════════════════════════════════════

        /// <summary>Opens the camera and starts the 15-fps preview loop.</summary>
        public void StartPreview()
        {
            if (_previewRunning) return;

            lock (_captureLock)
            {
                _capture = new VideoCapture(_cameraIndex);  // default backend (same as anomaly capture)
                if (!_capture.IsOpened())
                {
                    _logger.WriteErrorLog($"[VisualMonitor] StartPreview: camera {_cameraIndex} could not be opened.");
                    _capture.Dispose();
                    _capture = null;
                    return;
                }
                _capture.Set(VideoCaptureProperties.FrameWidth,  1280);
                _capture.Set(VideoCaptureProperties.FrameHeight, 800);
            }

            _previewRunning = true;
            _previewThread  = new Thread(PreviewLoop)
            {
                IsBackground = true,
                Name         = "VisualPreviewThread"
            };
            _previewThread.Start();
            _logger.WriteLog("[VisualMonitor] Preview started.");
        }

        /// <summary>Stops the preview loop and releases the camera.</summary>
        public void StopPreview()
        {
            _previewRunning = false;
            _previewThread?.Join(2000);

            lock (_captureLock)
            {
                _capture?.Dispose();
                _capture = null;
            }

            lock (_frameLock)
            {
                _latestFrame?.Dispose();
                _latestFrame = null;
            }

            _logger.WriteLog("[VisualMonitor] Preview stopped.");
        }

        private void PreviewLoop()
        {
            int intervalMs = 1000 / PreviewFps;

            while (_previewRunning)
            {
                try
                {
                    var frame = new Mat();
                    bool ok;
                    lock (_captureLock)
                    {
                        ok = _capture != null && _capture.Read(frame);
                    }

                    if (ok && !frame.Empty())
                    {
                        // Store for anomaly use
                        lock (_frameLock)
                        {
                            _latestFrame?.Dispose();
                            _latestFrame = frame.Clone();
                        }

                        // Fire event with a BGR→Bitmap conversion
                        try
                        {
                            // OV9281 is grayscale — convert to BGR for Bitmap
                            Mat display = frame;
                            bool isGray = frame.Channels() == 1;
                            if (isGray)
                            {
                                display = new Mat();
                                Cv2.CvtColor(frame, display, ColorConversionCodes.GRAY2BGR);
                            }

                            Bitmap bmp = BitmapConverter.ToBitmap(display);
                            if (isGray) display.Dispose();

                            FrameReady?.Invoke(this, bmp);
                        }
                        catch { /* best-effort UI event */ }
                    }

                    frame.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.WriteErrorLog($"[VisualMonitor] PreviewLoop error: {ex.Message}");
                }

                Thread.Sleep(intervalMs);
            }
        }

        // ══════════════════════════════════════════════════════════════════════════
        // Anomaly capture helpers
        // ══════════════════════════════════════════════════════════════════════════

        /// <summary>Returns JPEG bytes of the latest preview frame, or null.</summary>
        public Task<byte[]?> CaptureFrameAsync()
        {
            return Task.Run(() =>
            {
                lock (_frameLock)
                {
                    if (_latestFrame == null || _latestFrame.Empty())
                    {
                        _logger.WriteErrorLog("[VisualMonitor] CaptureFrame: no frame available yet.");
                        return null;
                    }
                    Cv2.ImEncode(".jpg", _latestFrame, out byte[] buf);
                    return (byte[]?)buf;
                }
            });
        }

        /// <summary>Records a 2-second MP4 clip from the shared camera. Returns temp path, caller deletes.</summary>
        public Task<string?> CaptureClipAsync()
        {
            return Task.Run(() =>
            {
                int width = 1280, height = 800;
                lock (_captureLock)
                {
                    if (_capture == null || !_capture.IsOpened()) return null;
                    int w = (int)_capture.Get(VideoCaptureProperties.FrameWidth);
                    int h = (int)_capture.Get(VideoCaptureProperties.FrameHeight);
                    if (w > 0 && h > 0) { width = w; height = h; }
                }

                string tempPath = Path.Combine(Path.GetTempPath(), $"visual_clip_{Guid.NewGuid():N}.mp4");

                using var writer = new VideoWriter(
                    tempPath, FourCC.MP4V, ClipFps,
                    new OpenCvSharp.Size(width, height), isColor: true);

                int totalFrames = ClipDurationSecs * ClipFps;
                using var frameBgr = new Mat();

                for (int i = 0; i < totalFrames; i++)
                {
                    var frame = new Mat();
                    bool ok;
                    lock (_captureLock) { ok = _capture != null && _capture.Read(frame); }

                    if (ok && !frame.Empty())
                    {
                        if (frame.Channels() == 1)
                            Cv2.CvtColor(frame, frameBgr, ColorConversionCodes.GRAY2BGR);
                        else
                            frame.CopyTo(frameBgr);

                        writer.Write(frameBgr);
                    }

                    frame.Dispose();
                    Thread.Sleep(1000 / ClipFps);
                }

                return (string?)tempPath;
            });
        }

        // ══════════════════════════════════════════════════════════════════════════
        // Public API methods
        // ══════════════════════════════════════════════════════════════════════════

        public async Task<VisualInspectionResponse?> CheckVisualImageAsync(Action stopMachineCallback)
        {
            try
            {
                byte[]? jpeg = await CaptureFrameAsync();
                if (jpeg == null) return null;

                using var content      = new MultipartFormDataContent();
                var       imageContent = new ByteArrayContent(jpeg);
                imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
                content.Add(imageContent, "file", "frame.jpg");

                HttpResponseMessage response = await _http.PostAsync(_imageEndpoint, content);
                response.EnsureSuccessStatusCode();

                string json   = await response.Content.ReadAsStringAsync();
                var    result = JsonSerializer.Deserialize<VisualInspectionResponse>(json);

                if (result != null && result.PlcStop)
                {
                    _logger.WriteLog($"[VisualMonitor] IMAGE anomaly → PLC stop. Trigger: {result.Trigger}");
                    stopMachineCallback?.Invoke();
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.WriteErrorLog($"[VisualMonitor] CheckVisualImageAsync: {ex.Message}");
                return null;
            }
        }

        public async Task<VisualInspectionResponse?> CheckVisualVideoAsync(Action stopMachineCallback)
        {
            string? clipPath = null;
            try
            {
                clipPath = await CaptureClipAsync();
                if (clipPath == null) return null;

                await using var fileStream   = File.OpenRead(clipPath);
                using var       content      = new MultipartFormDataContent();
                var             videoContent = new StreamContent(fileStream);
                videoContent.Headers.ContentType = MediaTypeHeaderValue.Parse("video/mp4");
                content.Add(videoContent, "file", "clip.mp4");

                HttpResponseMessage response = await _http.PostAsync(_videoEndpoint, content);
                response.EnsureSuccessStatusCode();

                string json   = await response.Content.ReadAsStringAsync();
                var    result = JsonSerializer.Deserialize<VisualInspectionResponse>(json);

                if (result != null && result.PlcStop)
                {
                    _logger.WriteLog($"[VisualMonitor] VIDEO anomaly → PLC stop. Trigger: {result.Trigger}");
                    stopMachineCallback?.Invoke();
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.WriteErrorLog($"[VisualMonitor] CheckVisualVideoAsync: {ex.Message}");
                return null;
            }
            finally
            {
                if (clipPath != null && File.Exists(clipPath))
                    try { File.Delete(clipPath); } catch { }
            }
        }

        // ══════════════════════════════════════════════════════════════════════════
        // IDisposable
        // ══════════════════════════════════════════════════════════════════════════

        public void Dispose()
        {
            if (_disposed) return;
            StopPreview();
            _http.Dispose();
            _disposed = true;
        }
    }
}
