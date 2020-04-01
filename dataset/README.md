# Instructions for Processing Data
This is the main entry point for collecting raw data and processing it into the RDC2 format needed for classification. 

## Experimental Setup
1. Have the Judo module set up in a large open space at about chest height. (~1.5m off the ground)
2. Set up the Judo radar to record data using the *VP1as* preset scan configuration.
3. Have multiple people run away from the radar at varying speeds. Do this many many many times. Make sure to have at least 2-3 people. The more the better.
3. Have multiple people run toward the radar at varying speeds. Do this many many many times. Make sure to have at least 2-3 people. The more the better.
4. Have multiple people ride bikes away form the radar at varying speeds. Do this many many many times. Also make sure to have at least 2-3 people. The more the better.
5. Have multiple people ride bikes toward the radar at varying speeds. Do this many many many times. Also make sure to have at least 2-3 people. The more the better.
6. Have someone drive a car away from the radar at varying speeds. 
7. Have someone drive a car toward the radar at varying speeds. 

## Data Capture
1. Save *running* scans to: */Borealis/data/raw/running*
2. Save *biking* scans to: */Borealis/data/raw/biking*
2. Save *driving* scans to: */Borealis/data/raw/driving*

## Data Processing
1. Run the script */Borealis/dataset/rdc1_to_rdc2.py*. This will save *.npy files in the */Borealis/data/processed* directory. Verify that all scans appear here.

## Data Sharing
1. Zip up the .npy files and send them to me. (Alternatively, commit them to Github -- YuanLiang has access).

### (Optional) Github commit instructions
1. *cd* into the */Borealis* directory
2. *git status* will show you what files can be added
3. *git add INSERT_FILENAME_YOU_WANT_TO_ADD* will stage the desired file for upload
4. *git commit -m "WRITE SOME MESSAGE HERE"* will commit the data to be added to Github
5. *git push origin master* will push the data up to the internet. I can get it from here.

## Key Things to Remember
1. The more data the better. More data = better classification models
2. Vary the movements when possible. Not everyone runs the same or rides a bike the same way. Varied data makes better classification models.
3. We're aiming to distinguish between running, biking and driving. Therefore, we need the same amount of running scnas, biking scans, and driving scans. 