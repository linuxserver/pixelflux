# Standard library imports
import asyncio
import ctypes
import http.server
import socketserver
import threading
import websockets
import os # For path operations

# pixelflux library imports
from pixelflux import CaptureSettings, ScreenCapture, StripeCallback

# --- Global "Context" Variables ---
# These are effectively global state or shared resources for the demo.
# Using a more explicit naming convention like 'g_' for clarity.
g_loop = None                   # The asyncio event loop, set in main()
g_capture_settings = None       # Capture settings, configured in main()
g_stripe_callback = None        # CTypes callback, created in main()
g_module = None                 # ScreenCapture instance, created in main()

g_clients = set()               # Set of connected WebSocket clients
g_is_capturing = False          # Flag indicating if capture is active
g_h264_stripe_queue = None      # asyncio.Queue for H.264 stripes, created when capture starts
g_send_task = None              # asyncio.Task for broadcasting stripes
# --- End Global Variables ---


async def send_h264_stripes():
    """
    Continuously retrieves H.264 stripe data from the global queue
    and broadcasts it to all connected WebSocket clients.
    """
    global g_h264_stripe_queue, g_clients
    print("Stripe sending task started.")
    try:
        while True:
            if g_h264_stripe_queue is None: # Should not happen if task is managed correctly
                await asyncio.sleep(0.1)
                continue

            h264_bytes_with_prefix = await g_h264_stripe_queue.get()
            
            if not g_clients:
                g_h264_stripe_queue.task_done()
                continue
            
            # Create a list of tasks to send to current clients
            active_clients = list(g_clients) # Avoid issues if set modified during iteration
            tasks = [client.send(h264_bytes_with_prefix) for client in active_clients]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            g_h264_stripe_queue.task_done()
    except asyncio.CancelledError:
        print("Stripe sending task cancelled.")
        raise
    except Exception as e:
        print(f"Error in send_h264_stripes: {e}")
    finally:
        print("Stripe sending task finished.")


async def ws_handler(websocket, path=None):
    """
    Handles new WebSocket connections, manages client set, and controls capture state.
    """
    global g_clients, g_is_capturing, g_h264_stripe_queue, g_module, g_send_task
    global g_capture_settings, g_stripe_callback # These are set in main()

    g_clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}. Total clients: {len(g_clients)}")

    if not g_is_capturing and g_module:
        print("First client connected. Starting H.264 capture...")
        g_h264_stripe_queue = asyncio.Queue() # Fresh queue for the new capture session

        if g_capture_settings is None or g_stripe_callback is None:
            print("Error: Critical server components (settings/callback) not initialized.")
            await websocket.close(code=1011, reason="Server configuration error")
            g_clients.remove(websocket)
            return

        g_module.start_capture(g_capture_settings, g_stripe_callback)
        g_is_capturing = True
        
        if g_send_task is None or g_send_task.done():
            g_send_task = asyncio.create_task(send_h264_stripes())
        print("H.264 capture process initiated.")
    elif not g_module:
        print("Error: Pixelflux module not initialized. Cannot start capture.")
        try:
            await websocket.send("ERROR: Server-side capture module not ready.")
            await websocket.close(code=1011, reason="Server module error")
        except websockets.exceptions.ConnectionClosed:
            pass
        if websocket in g_clients: g_clients.remove(websocket)
        return

    try:
        async for _ in websocket:
            # This demo doesn't expect messages from the client
            pass
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client {websocket.remote_address} disconnected: Code {e.code}, Reason: '{e.reason or 'N/A'}'")
    except Exception as e:
        print(f"WebSocket error for client {websocket.remote_address}: {e}")
    finally:
        if websocket in g_clients:
            g_clients.remove(websocket)
        print(f"Client connection closed: {websocket.remote_address}. Remaining clients: {len(g_clients)}")

        if g_is_capturing and not g_clients and g_module: # If last client disconnects
            print("Last client disconnected. Stopping H.264 capture...")
            g_module.stop_capture()
            g_is_capturing = False
            if g_send_task and not g_send_task.done():
                g_send_task.cancel()
                # The task will be awaited in the main finally block if needed
                g_send_task = None 
            print("H.264 capture process stopped.")
            # g_h264_stripe_queue can be left for GC or explicitly set to None
            g_h264_stripe_queue = None


def py_stripe_callback(result_ptr, user_data):
    """
    Callback function invoked by the pixelflux C++ module from a separate thread
    when an H.264 encoded stripe is ready.
    """
    global g_is_capturing, g_h264_stripe_queue, g_loop 

    if g_is_capturing and result_ptr and g_h264_stripe_queue is not None:
        result = result_ptr.contents
        if result.data and result.size > 0:
            data_bytes_ptr = ctypes.cast(
                result.data, ctypes.POINTER(ctypes.c_ubyte * result.size)
            )
            h264_stripe_with_prefix = bytes(data_bytes_ptr.contents)
            
            if g_loop and not g_loop.is_closed(): 
                asyncio.run_coroutine_threadsafe(
                    g_h264_stripe_queue.put(h264_stripe_with_prefix), 
                    g_loop
                )
    # Memory for `result.data` is managed by the C++ module.

def start_http_server(port=9001):
    """
    Starts a simple HTTP server in a new thread to serve client-side files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    class DirectoryServingHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=script_dir, **kwargs)
        def log_message(self, format, *args):
            pass # Quieter HTTP logs

    with socketserver.TCPServer(("localhost", port), DirectoryServingHTTPRequestHandler) as httpd:
        print(f"HTTP server serving files from '{os.path.abspath(script_dir)}' on http://localhost:{port}/")
        print(f"  (Open http://localhost:{port}/index.html in browser to connect to WebSocket)")
        httpd.serve_forever()

async def main_async_routine():
    """
    Main asynchronous routine to set up and run the server.
    """
    global g_loop, g_capture_settings, g_stripe_callback, g_module
    global g_is_capturing, g_send_task # For cleanup

    # asyncio.run() creates and manages the loop. Get a reference for the C callback.
    g_loop = asyncio.get_running_loop()

    # --- Configure Screen Capture Parameters ---
    g_capture_settings = CaptureSettings()
    g_capture_settings.capture_width = 1920
    g_capture_settings.capture_height = 1080
    g_capture_settings.capture_x = 0
    g_capture_settings.capture_y = 0
    g_capture_settings.target_fps = 60.0
    g_capture_settings.output_mode = 1 # 1 for H.264
    g_capture_settings.h264_crf = 25
    g_capture_settings.use_paint_over_quality = True 
    g_capture_settings.paint_over_trigger_frames = 2 
    g_capture_settings.damage_block_threshold = 15   
    g_capture_settings.damage_block_duration = 30
    g_capture_settings.h264_fullcolor = True
    # --- End Capture Configuration ---

    g_stripe_callback = StripeCallback(py_stripe_callback)
    g_module = ScreenCapture()

    ws_server_instance = None # To hold the websockets server object

    if not g_module:
        print("Fatal: Failed to initialize pixelflux ScreenCapture module.")
        return

    print("Pixelflux capture module initialized.")
    
    # Start HTTP server in a daemon thread.
    http_thread = threading.Thread(target=start_http_server, args=(9001,), daemon=True)
    http_thread.start()

    ws_port = 9000
    shutdown_event = asyncio.Event() # Used to keep server running until interrupt

    try:
        # websockets.serve should use the currently running loop (managed by asyncio.run())
        # No explicit `loop=` argument is typically needed here with asyncio.run()
        ws_server_instance = await websockets.serve(
            ws_handler,
            'localhost',
            ws_port,
            compression=None
        )
        print(f"WebSocket server started (streaming H.264) on ws://localhost:{ws_port}")
        print("Waiting for client connections... Press Ctrl+C to stop.")
        
        await shutdown_event.wait() # Keep the server running until event is set (by KeyboardInterrupt)

    except OSError as e: 
        print(f"Failed to start server (e.g., port in use?): {e}")
    # KeyboardInterrupt will be handled by the outer asyncio.run() wrapper,
    # or we can catch it here to set the shutdown_event.
    except KeyboardInterrupt: # This allows graceful shutdown of this coroutine
        print("\nKeyboardInterrupt caught in main_async_routine. Signaling shutdown...")
    finally:
        print("Cleaning up resources in main_async_routine...")
        if g_is_capturing and g_module:
            g_module.stop_capture() # Ensure capture is stopped
            g_is_capturing = False # Update flag
            print("Capture stopped.")

        if ws_server_instance:
            ws_server_instance.close()
            await ws_server_instance.wait_closed()
            print("WebSocket server closed.")

        if g_send_task and not g_send_task.done():
            g_send_task.cancel()
            try:
                await g_send_task 
            except asyncio.CancelledError:
                pass # Expected
            print("Stripe sending task cancelled and awaited.")
        
        if g_module:
            del g_module # Allow C++ destructor to run if applicable
            g_module = None
            print("Capture module resources released.")
        
        # asyncio.run() handles loop cleanup and cancellation of remaining tasks.
        print("Async routine cleanup finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main_async_routine())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it occurs outside the main_async_routine's try/except,
        # or if main_async_routine re-raises it.
        print("\nApplication shutdown initiated by KeyboardInterrupt (at top level).")
    except Exception as e:
        print(f"Unhandled exception in __main__: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application exiting.")
