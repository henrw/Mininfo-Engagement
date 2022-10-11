from genericpath import isdir
import click
from pathlib import Path
import moviepy.editor as mp
import time
@click.command()
@click.option('--output_dir', '-o', type=click.Path(exists=False), help='Output directory of audio file(s)')
@click.argument('input')

            
def main(output_dir, input):
    if output_dir is None:
        output_dir = Path("data/audio/")
    output_dir.mkdir(exist_ok=True)

    input = Path(input)
    assert(Path.exists(input))
    if Path.is_dir(input):
        input_dir = input
        file_list = list(input_dir.iterdir())
    else:
        file_list = [output_dir]
    
    print(f"{len(file_list)} videos to convert...",end="\n\n")
    for file in file_list:
        start_time = time.time()
        mp.VideoFileClip(str(file)).audio.write_audiofile(output_dir / file.with_suffix(".wav").name)
        print(f"Time: {time.time() - start_time}")
        print("---")


if __name__ == "__main__":
    main()
