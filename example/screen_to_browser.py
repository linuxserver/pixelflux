# -*- coding: utf-8 -*-
"""
A multi-client WebSocket and HTTP server for streaming screen captures.

This script demonstrates the pixelflux library's instance-safe capabilities.
It can handle multiple WebSocket clients, each with its own independent
screen capture session. The capture region can be controlled via the URL hash.
"""

# Standard library imports
import asyncio
import os
import mimetypes
import websockets
import websockets.asyncio.server as ws_async
import threading

# Third-party library imports
from pixelflux import CaptureSettings, ScreenCapture, StripeCallback

# ==============================================================================
# --- BASE CONFIGURATION SETTINGS ---
# These settings will be used as a template for each new connection.
# Modify the parameters below to test different capture and encoding options.
# ==============================================================================
HTTP_PORT = 9001
WS_PORT = 9000

# Create a default template for capture settings.
base_capture_settings = CaptureSettings()

# --- Debugging ---
# Enable/disable the continuous FPS and settings log printed to the console.
base_capture_settings.debug_logging = True

# --- Core Capture ---
base_capture_settings.capture_width = 1920
base_capture_settings.capture_height = 1080
base_capture_settings.capture_x = 0  # This can be overridden by the URL
base_capture_settings.capture_y = 0
base_capture_settings.target_fps = 60.0
base_capture_settings.capture_cursor = False

# --- Encoding Mode ---
# Sets the output codec. 0 for JPEG, 1 for H.264.
base_capture_settings.output_mode = 1
# Force CPU encoding and ignore hardware encoders
base_capture_settings.use_cpu = False

# --- H.264 Quality Settings ---
# Constant Rate Factor (0-51, lower is better quality & higher bitrate).
# Good values are typically 18-28.
base_capture_settings.h264_crf = 25
# CRF for H.264 paintover on static content. Used if lower (better) than h264_crf.
base_capture_settings.h264_paintover_crf = 18
# Number of high-quality H.264 frames to send in a burst when a paintover is triggered.
base_capture_settings.h264_paintover_burst_frames = 5
# Use I444 (full color) instead of I420. Better quality, higher CPU/bandwidth.
base_capture_settings.h264_fullcolor = False
# Encode full frames instead of just changed stripes.
base_capture_settings.h264_fullframe = False
# Flag the stream to be in streaming mode to bypass all vnc logic
base_capture_settings.h264_streaming_mode = False
# Pass a vaapi node index 0 = renderD128, -1 to disable
base_capture_settings.vaapi_render_node_index = -1
# Switches to CBR mode and ignores CRF value. Used in conjunction with h264_bitrate_kbps.
base_capture_settings.h264_cbr_mode = False
# Target bitrate in kbps for CBR mode. Required when h264_cbr_mode is enabled.
base_capture_settings.h264_bitrate_kbps = 4000
# Optional VBV buffer size in kilobits for custom buffer size.
base_capture_settings.h264_vbv_buffer_size_kb = 400
# Allow pixelflux to adjust its capture width and height. Overrides provided width and height when enabled.
base_capture_settings.auto_adjust_screen_capture_size = True
#

# --- Change Detection & Optimization ---
# Use a higher quality setting for static regions that haven't changed for a while.
base_capture_settings.use_paint_over_quality = True
# Number of frames of no motion in a stripe to trigger a high-quality "paint-over".
base_capture_settings.paint_over_trigger_frames = 15
# Consecutive changes to a stripe to trigger a "damaged" state (uses base quality).
base_capture_settings.damage_block_threshold = 10
# Number of frames a stripe stays "damaged" after being triggered.
base_capture_settings.damage_block_duration = 30

# --- JPEG Quality Settings ---
# Quality of jpegs under motion
base_capture_settings.jpeg_quality = 40
# Quality of jpegs on static content paintovers
base_capture_settings.paint_over_jpeg_quality = 90

# --- Watermarking ---
# The path MUST be a byte string (b"") and point to a valid PNG file.
#base_capture_settings.watermark_path = b"/path/to/image.png"
# Sets the watermark location on the screen. Default is 0 (disabled).
# Options: 0:None, 1:TopLeft, 2:TopRight, 3:BottomLeft, 4:BottomRight, 5:Middle, 6:Animated
base_capture_settings.watermark_location_enum = 0

# ==============================================================================
# --- Multi-Client State Management ---
# ==============================================================================
g_loop = None  # The main asyncio event loop.

# This dictionary holds the state for each active client.
# The key is the WebSocket connection object.
# The value is another dictionary containing the client's capture module, queue, and task.
ACTIVE_CLIENTS = {}
CLIENT_LOCK = threading.Lock() # Lock for thread-safe modifications to ACTIVE_CLIENTS.

async def send_stripes_task(websocket, queue):
    """
    Pulls video stripes from a client-specific queue and sends them.
    This task is cancelled when the client disconnects.
    """
    print(f"Send task started for client {websocket.remote_address}.")
    try:
        # This loop will run until the connection is closed,
        # which will raise a ConnectionClosed exception.
        while True:
            data_to_send = await queue.get()
            await websocket.send(data_to_send)
            queue.task_done()

    except websockets.exceptions.ConnectionClosed:
        # This is the expected, clean way to exit the loop when a client disconnects.
        print(f"Connection closed for {websocket.remote_address}. Send task stopping.")
    
    except asyncio.CancelledError:
        # This happens when the main handler cancels us during cleanup.
        print(f"Send task was cancelled for {websocket.remote_address}.")

    except Exception as e:
        # Catch any other unexpected errors.
        print(f"[ERROR] Send task for client {websocket.remote_address} failed unexpectedly: {e}")
    
    finally:
        print(f"Send task for {websocket.remote_address} has finished.")

async def websocket_handler(websocket):
    """
    Manages a single WebSocket connection and its dedicated screen capture lifecycle.
    """
    path = websocket.request.path
    client_id = id(websocket)
    print(f"New client connected from {websocket.remote_address} with path '{path}' (ID: {client_id}).")

    client_module = None
    send_task = None
    # Keep a reference to the callback object to prevent it from being garbage collected
    c_callback = None

    try:
        # --- 1. Configure Capture for this Specific Client ---
        client_settings = base_capture_settings
        try:
            x_offset = int(path.strip('/'))
            client_settings.capture_x = x_offset
            print(f"Client {client_id} requested custom capture at x={x_offset}.")
        except (ValueError, TypeError):
            print(f"Client {client_id} using default capture at x=0.")
            client_settings.capture_x = 0

        # --- 2. Create Resources for this Client ---
        client_module = ScreenCapture()
        client_queue = asyncio.Queue(maxsize=120)

        # --- 3. Create a unique callback (closure) for this client ---
        # This function "closes over" client_queue and g_loop, giving it access
        # without needing global lookups or user_data.
        def client_specific_callback(result_ptr, user_data_ptr):
            """Callback invoked by pixelflux when a new video stripe is ready."""
            if result_ptr:
                result = result_ptr.contents
                if result.size > 0 and g_loop and not g_loop.is_closed():
                    raw_data_from_cpp = bytes(result.data[:result.size])
                    final_payload = raw_data_from_cpp
                    
                    if client_settings.output_mode == 0:
                        final_payload = b"\x03\x00" + raw_data_from_cpp
                    
                    asyncio.run_coroutine_threadsafe(
                        client_queue.put(final_payload), g_loop
                    )
        
        # Convert the Python closure into a C-compatible function pointer
        c_callback = StripeCallback(client_specific_callback)

        # --- 4. Register and Start Resources for this Client ---
        send_task = asyncio.create_task(send_stripes_task(websocket, client_queue))
        ACTIVE_CLIENTS[websocket] = {
            "module": client_module,
            "queue": client_queue,
            "task": send_task,
            "callback": c_callback # Store reference to prevent GC
        }

        # --- 5. Start the Capture with the correct 3 arguments ---
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, client_module.start_capture, client_settings, c_callback
        )
        print(f"Capture started for client {client_id}.")

        # --- 6. Wait for the Client to Disconnect ---
        async for _ in websocket:
            pass # Keep the connection alive

    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_id} disconnected normally.")
    except Exception as e:
        print(f"[ERROR] WebSocket handler for client {client_id} error: {e}")
    finally:
        # --- 7. Clean Up Resources for this Specific Client ---
        print(f"Cleaning up resources for client {client_id}...")
        
        if send_task and not send_task.done():
            send_task.cancel()
            try: await send_task
            except asyncio.CancelledError: pass
        
        if client_module:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, client_module.stop_capture)

        ACTIVE_CLIENTS.pop(websocket, None)
        print(f"Cleanup complete for client {client_id}. Active clients: {len(ACTIVE_CLIENTS)}")

async def handle_http_request(reader, writer):
    """Handle HTTP requests by serving static files from the script directory."""
    try:
        request_line = await reader.readline()
        if not request_line:
            return

        parts = request_line.split()
        if len(parts) < 2 or parts[0] != b'GET':
            writer.write(b'HTTP/1.1 405 Method Not Allowed\r\n\r\n')
            return

        path = parts[1].decode().split('#')[0] # Ignore hash part
        if path == '/':
            path = '/index.html'

        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, path.lstrip('/'))
        
        # Security check to prevent directory traversal attacks
        if not os.path.realpath(full_path).startswith(os.path.realpath(script_dir)):
            writer.write(b'HTTP/1.1 403 Forbidden\r\n\r\n')
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
    finally:
        if not writer.is_closing():
            try:
                await writer.drain()
            except ConnectionResetError:
                pass # Client closed connection before we could finish.
            finally:
                writer.close()


async def main():
    """Initializes and starts the WebSocket and HTTP servers."""
    global g_loop
    g_loop = asyncio.get_running_loop()

    http_server = await asyncio.start_server(handle_http_request, 'localhost', HTTP_PORT)
    print(f"HTTP server serving on http://localhost:{HTTP_PORT}/")
    print(f"-> Open http://localhost:{HTTP_PORT}/ to start a capture at (0,0).")
    print(f"-> Open http://localhost:{HTTP_PORT}/#10 to start a capture at (10,0).")

    ws_server = None
    try:
        ws_server = await ws_async.serve(websocket_handler, 'localhost', WS_PORT, compression=None)
        print(f"WebSocket server started on ws://localhost:{WS_PORT}")
        print("Waiting for client connections... Press Ctrl+C to stop.")
        await asyncio.Event().wait()
    except OSError as e:
        print(f"[FATAL] Could not start server (is port {WS_PORT} in use?): {e}")
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        print("Shutting down all client connections...")
        # Create a list of cleanup tasks for all active clients
        cleanup_tasks = []
        with CLIENT_LOCK:
            clients_to_clean = list(ACTIVE_CLIENTS.keys())
        
        for ws in clients_to_clean:
            # Closing the websocket connection will trigger its handler's finally block
            cleanup_tasks.append(ws.close(code=1001, reason='Server shutting down'))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        if ws_server:
            ws_server.close()
            await ws_server.wait_closed()
        
        http_server.close()
        await http_server.wait_closed()
        print("All servers and connections closed. Goodbye.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication exiting.")
