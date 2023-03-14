# demand-layaring
Darknet memory optimization

1. Download .weigts files (already apply the batch normalization)
> https://drive.google.com/drive/folders/1xFJ4G4WS8wzsQ7rsO6HihxBV-uGqMvGG?usp=sharing

2. Download test50.mp4 files

3. Edit Make file (Choose one pipeline architecture, and change 0 to 1)

- sequential = 0
- sync = 0
- async = 0
- two_stage = 0
- trade_off =0 (Only async and two_stage)

4. Excute .sh file
