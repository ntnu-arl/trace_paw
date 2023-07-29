import image, network, omv, rpc, sensor, struct

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.HQVGA)
sensor.skip_frames(time = 2000)


omv.disable_fb(True)

interface = rpc.rpc_usb_vcp_slave()

def jpeg_image_snapshot(data):
    pixformat, framesize = bytes(data).decode().split(",")
    sensor.set_pixformat(eval(pixformat))
    sensor.set_framesize(eval(framesize))
    # img = sensor.snapshot().compress(quality=90)
    img = sensor.snapshot()
    img = img.scale(x_size = 45, y_size = 30)
    return struct.pack("<I", img.size())

def jpeg_image_read_cb():
    interface.put_bytes(sensor.get_fb().bytearray(), 5000) # timeout


# Read data from the frame buffer given a offset and size.
# If data is empty then a transfer is scheduled after the RPC call finishes.
#
# data is a 4 byte size and 4 byte offset.
def jpeg_image_read(data):
    if not len(data):
        interface.schedule_callback(jpeg_image_read_cb)
        return bytes()
    else:
        offset, size = struct.unpack("<II", data)
        return memoryview(sensor.get_fb().bytearray())[offset:offset+size]

# Register call backs.

interface.register_callback(jpeg_image_snapshot)
interface.register_callback(jpeg_image_read)

# Once all call backs have been registered we can start
# processing remote events. interface.loop() does not return.

interface.loop()
