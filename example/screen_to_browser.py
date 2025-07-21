# -*- coding: utf-8 -*-
"""
A single-client WebSocket and HTTP server for streaming screen captures.

This script is a demonstration of the pixelflux library. It captures the screen
and sends it to a single connected WebSocket client. All capture settings can be
configured in the 'CONFIGURATION SETTINGS' block below.
"""

# Standard library imports
import asyncio
import os
import mimetypes
import websockets

# Third-party library imports
from pixelflux import CaptureSettings, ScreenCapture

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# Modify the parameters below to test different capture and encoding options.
# ==============================================================================
HTTP_PORT = 9001
WS_PORT = 9000

capture_settings = CaptureSettings()

# --- Core Capture ---
capture_settings.capture_width = 1920
capture_settings.capture_height = 1080
capture_settings.capture_x = 0
capture_settings.capture_y = 0
capture_settings.target_fps = 60.0
capture_settings.capture_cursor = False

# --- Encoding Mode ---
# Sets the output codec. 0 for JPEG, 1 for H.264.
capture_settings.output_mode = 1

# --- H.264 Quality Settings ---
# Constant Rate Factor (0-51, lower is better quality & higher bitrate).
# Good values are typically 18-28.
capture_settings.h264_crf = 25
# Use I444 (full color) instead of I420. Better quality, higher CPU/bandwidth.
capture_settings.h264_fullcolor = False
# Encode full frames instead of just changed stripes.
capture_settings.h264_fullframe = False
# Pass a vaapi node index 0 = renderD128, -1 to disable
capture_settings.vaapi_render_node_index = -1

# --- Change Detection & Optimization ---
# Use a higher quality setting for static regions that haven't changed for a while.
capture_settings.use_paint_over_quality = True
# Number of frames of no motion in a stripe to trigger a high-quality "paint-over".
capture_settings.paint_over_trigger_frames = 15
# Consecutive changes to a stripe to trigger a "damaged" state (uses base quality).
capture_settings.damage_block_threshold = 10
# Number of frames a stripe stays "damaged" after being triggered.
capture_settings.damage_block_duration = 30

# --- Watermarking ---
# The path MUST be a byte string (b"") and point to a valid PNG file.
#capture_settings.watermark_path = b"/path/to/image.png"

# Sets the watermark location on the screen. Default is 0 (disabled).
# Options: 0:None, 1:TopLeft, 2:TopRight, 3:BottomLeft, 4:BottomRight, 5:Middle, 6:Animated
capture_settings.watermark_location_enum = 0

# ==============================================================================
# --- Global State ---
# ==============================================================================
g_loop = None                   # The main asyncio event loop.
g_module = None                 # The ScreenCapture instance.
g_active_client = None          # Holds the single active WebSocket client.
g_is_capturing = False          # Flag indicating if capture is active.
g_h264_stripe_queue = None      # asyncio.Queue for H.264 stripes.
g_send_task = None              # asyncio.Task for sending stripes.

g_cleanup_lock = None
g_is_shutting_down = False

async def cleanup():
    """A single, race-proof function to shut down all capture resources."""
    global g_is_shutting_down, g_is_capturing, g_module, g_send_task, g_active_client, g_h264_stripe_queue, g_cleanup_lock
    
    # Initialize lock if not already done
    if g_cleanup_lock is None:
        g_cleanup_lock = asyncio.Lock()
    
    async with g_cleanup_lock:
        if g_is_shutting_down:
            return
        
        print("Cleanup initiated...")
        g_is_shutting_down = True
        
        # Stop capturing first
        if g_is_capturing:
            g_is_capturing = False
        
        # Cancel the send task
        if g_send_task and not g_send_task.done():
            g_send_task.cancel()
            try:
                await g_send_task
            except asyncio.CancelledError:
                pass
        
        # Clear the queue to prevent memory buildup
        if g_h264_stripe_queue:
            while not g_h264_stripe_queue.empty():
                try:
                    g_h264_stripe_queue.get_nowait()
                    g_h264_stripe_queue.task_done()
                except asyncio.QueueEmpty:
                    break
        
        # Stop the capture module
        if g_module:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, g_module.stop_capture)
            except Exception as e:
                print(f"[WARNING] Error stopping capture: {e}")
        
        # Clear references
        g_active_client = None
        g_send_task = None
        g_h264_stripe_queue = None
        g_is_shutting_down = False
        print("Cleanup complete.")

async def send_h264_stripes():
    """Retrieves H.264 stripes from the queue and sends them to the active client."""
    global g_h264_stripe_queue, g_active_client
    try:
        while True:
            h264_bytes = await g_h264_stripe_queue.get()
            if g_active_client:
                try:
                    await g_active_client.send(h264_bytes)
                except websockets.exceptions.ConnectionClosed:
                    pass  # Client disconnected, main handler will clean up.
            g_h264_stripe_queue.task_done()
    except asyncio.CancelledError:
        pass  # Expected way for the task to be stopped.
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the send task: {e}")
    finally:
        print("Stripe sending task has stopped.")


async def websocket_handler(websocket, path=None):
    """Manages a single WebSocket connection and the screen capture lifecycle."""
    global g_active_client, g_is_capturing, g_h264_stripe_queue, g_module, g_send_task

    # Handle existing client - clean up first, then accept new connection
    if g_active_client is not None:
        print("New connection detected. Cleaning up previous connection...")
        await cleanup()
        # Small delay to ensure cleanup is complete
        await asyncio.sleep(0.1)

    g_active_client = websocket
    print("Client connected. Starting screen capture...")

    try:
        # Ensure we have a fresh queue
        g_h264_stripe_queue = asyncio.Queue(maxsize=120)
        
        # Start capture with error handling
        try:
            g_module.start_capture(capture_settings, stripe_callback_handler)
            g_is_capturing = True
            g_send_task = asyncio.create_task(send_h264_stripes())
            print("Screen capture and stream started.")
        except Exception as e:
            print(f"[ERROR] Failed to start capture: {e}")
            await websocket.close(code=1011, reason="Failed to start capture")
            return

        # Keep connection alive and handle messages
        async for message in websocket:
            # Echo back any messages (for debugging/keepalive)
            if isinstance(message, str) and message == "ping":
                await websocket.send("pong")
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected normally.")
    except Exception as e:
        print(f"[ERROR] WebSocket handler error: {e}")
    finally:
        # Only cleanup if this websocket is still the active client
        if websocket is g_active_client:
            await cleanup()

def stripe_callback_handler(result):
    """Callback invoked by pixelflux when a new video stripe is ready."""
    try:
        if g_is_capturing and result and g_h264_stripe_queue is not None:
            data = result.data
            if data and data.nbytes > 0:
                # The result object contains the encoded video data
                if g_loop and not g_loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            g_h264_stripe_queue.put(bytes(data)), g_loop
                        )
                    except RuntimeError:
                        # Loop might be closed, ignore
                        pass
    except Exception as e:
        # Silently handle any callback errors to prevent crashes
        print(f"[WARNING] Error in stripe callback: {e}")

async def handle_http_request(reader, writer):
    """Handle HTTP requests by serving static files from the script directory."""
    try:
        request_line = await reader.readline()
        if not request_line:
            return
            
        parts = request_line.split()
        if len(parts) < 2 or parts[0] != b'GET':
            writer.write(b'HTTP/1.1 405 Method Not Allowed\r\n\r\n')
            await writer.drain()
            writer.close()
            return
            
        path = parts[1].decode()
        if path == '/':
            path = '/index.html'
            
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, path.lstrip('/'))
        
        # Security check: prevent directory traversal
        if not full_path.startswith(script_dir):
            writer.write(b'HTTP/1.1 403 Forbidden\r\n\r\n')
            await writer.drain()
            writer.close()
            return
            
        if os.path.isfile(full_path):
            with open(full_path, 'rb') as f:
                content = f.read()
                
            content_type = mimetypes.guess_type(full_path)[0] or 'application/octet-stream'
            
            headers = f'HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(content)}\r\n\r\n'
            writer.write(headers.encode())
            writer.write(content)
        else:
            writer.write(b'HTTP/1.1 404 Not Found\r\n\r\n')
            
    except Exception as e:
        print(f"[HTTP Error] {e}")
        writer.write(b'HTTP/1.1 500 Internal Server Error\r\n\r\n')
    finally:
        await writer.drain()
        writer.close()

async def main():
    """Initializes resources and starts the WebSocket and HTTP servers."""
    global g_loop, g_module, g_is_capturing, g_send_task

    g_loop = asyncio.get_running_loop()

    g_module = ScreenCapture()
    if not g_module:
        print("[FATAL] Failed to initialize pixelflux ScreenCapture module.")
        return
    print("Pixelflux capture module initialized.")

    # Start HTTP server using asyncio
    http_server = await asyncio.start_server(
        handle_http_request, 'localhost', HTTP_PORT
    )
    print(f"HTTP server is serving files from current directory")
    print(f"-> Open http://localhost:{HTTP_PORT}/index.html in your browser.")

    ws_server = None
    try:
        ws_server = await websockets.serve(
            websocket_handler, 'localhost', WS_PORT, compression=None
        )
        print(f"WebSocket server started on ws://localhost:{WS_PORT}")
        print("Waiting for a client connection... Press Ctrl+C to stop.")
        await asyncio.Event().wait()
    except OSError as e:
        print(f"[FATAL] Could not start server (is port {WS_PORT} in use?): {e}")
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        await cleanup()
        if ws_server:
            ws_server.close()
            await ws_server.wait_closed()
        if http_server:
            http_server.close()
            await http_server.wait_closed()
        print("Cleanup complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication exiting.")
