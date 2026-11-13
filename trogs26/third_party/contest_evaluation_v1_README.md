# TROGS-26 Evaluation Code

Code released in conjunction with the 2026 ICDAR Competition in 
Text Recognition on Greek Squeezes (TROGS-26)

## Usage 
Call `run_evaluations(pred_dir,gt_dir)` to evaluate.

`pred_dir` should contain your method's transcription files
* Filenames should be of the form `{squeeze}_transcript.txt`
* Contents are line by line character transcriptions
* Characters may be either Latin proxies or UTF-8 Greek
* Whitespace besides newlines is ignored
* Letter case is ignored
* No punctuation should be included

`gt_dir` should contain letter box files with ground truth
* Filenames should be of the form `{squeeze}_Rotation{1|2}_300dpi_letters.txt`
* Ground truth files for training/validation can be downloaded at https://scholarworks.smith.edu/dds_data/18/

## Details

To sidestep the challenging problem of document layout analysis, this contest treats all squeezes as simple single-column text.  Each line of characters in the squeeze should appear as its own line in the transcript, reading from left to right.  Tabs and spaces will be ignored, but newline characters are retained.  Punctuation should not be included.

Each squeeze is scored by counting the edit distance between the proposed transcript and the ground truth.  The number of ground truth characters is also recorded.  The final score is the total number of errors summed over all squeezes, divided by the total number of characters in all ground truth.

## License

[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Nicholas R. Howe