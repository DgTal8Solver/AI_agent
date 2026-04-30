from system.audio_io import AudioCapture, AudioPlayer
from sounddevice import query_devices

def main():
    devices = query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    output_devices = [device for device in devices if device['max_output_channels'] > 0]
    
    print("\n".join(f"{i}. {device['name']} | {device['index']}" for i, device in enumerate(input_devices)))
    in_device = input_devices[int(input("Input Device: "))]["index"]
    print("---------------------------")
    print("\n".join(f"{i}. {device['name']} | {device['index']}" for i, device in enumerate(output_devices)))
    out_device = output_devices[int(input("Output Device: "))]["index"]

    params = dict(
        samplerate = 20100,
        blocksize = 256,
        channels = 1
    )
    capture = AudioCapture(**params, device = in_device)
    player = AudioPlayer(**params, device = out_device)

    capture.start()
    player.start()

    while True:
        try:
            chunk = capture.get_chunk()
            if chunk is not None:
                player.put_chunk(chunk)
        except KeyboardInterrupt:
            print("Stop")
            capture.stop()
            player.stop()
            break

if __name__ == "__main__":
    main()