convert:
    ffmpeg -i  ~/Movies/GoPro/StartTakeOutForDetection/4k60/GX010473.MP4 -c:v libx264 -crf 18 -preset veryfast -c:a copy input_1_h264.mp4;

converts:
    ls ~/Movies/GoPro/StartTakeOutForDetection/4k60/GX*0555.MP4 | sort | sed "s/^/file '/;s/$/'/"  > files.txt;
    ls ~/Movies/GoPro/StartTakeOutForDetection2/GH/G*.MP4 | sort | sed "s/^/file '/;s/$/'/"  > filesGH.txt;
    ls ~/Movies/GoPro/StartTakeOutForDetection2/GX/G*.MP4 | sort | sed "s/^/file '/;s/$/'/"  > filesGX.txt;
    ffmpeg -f concat -safe 0 -i filesGH.txt -c copy GH_all_merged.mp4;
    ffmpeg -f concat -safe 0 -i filesGX.txt -c copy GX_all_merged.mp4;
    ffmpeg -i GX_27k_merged.mp4 -c:v libx264 -crf 18 -preset veryfast -c:a copy GX_27k_merged_h264.mp4;
    ffmpeg -i GH_all_merged.mp4 -c:v libx264 -crf 18 -preset veryfast -c:a copy GH_all_merged_h264.mp4;
    ffmpeg -i GX_all_merged.mp4 -c:v libx264 -crf 18 -preset veryfast -c:a copy GX_all_merged_h264.mp4;
    ffmpeg -i GX_1080_merged.mp4 -c:v libx264 -crf 18 -preset veryfast -c:a copy GX_1080_merged_h264.mp4;

cal:
    python3 main.py --calibrate_line

run: python main.py \
  --video GH_all_merged_h264.mp4 \
  --gallery ./gallery \
  --out_video out.mp4 \
  --out_csv laps.csv \
  --line "50%,5%,50%,95%" \
  --line_width 20 \
  --digits_has_plate



