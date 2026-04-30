from aiagent.system.audio_io import AudioCapture, AudioPlayer

def main():
    params = dict(
        samplerate = 24000,
        blocksize = 256,
        channels = 2,
        dtype = 'float16'
    )
    capture = AudioCapture(**params)
    player = AudioPlayer(**params)

    capture.start()
    player.start()

    while True:
        try:
            chunk = capture.get_chunk()
            if chunk:
                player.put_chunk(chunk)
        except KeyboardInterrupt:
            print("Stop")
            capture.stop()
            player.stop()
            break

if __name__ == "__main__":
    main()