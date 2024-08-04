DROPBOX_LINK="https://www.dropbox.com/scl/fo/cxqy4z5k33gnyhz455o2u/ABjCwsklQiwzYde5pkcf2RE?e=1&preview=ttv_detection_data.npz&rlkey=tvgxuu0om9i8ozmhm4uvwmzhb&st=tqpet46a&dl=0"
FILE_NAME="ttv_detection_data.npz"

echo "Downloading dataset from Dropbox..."
curl -L -o ${FILE_NAME} "${DROPBOX_LINK}"

echo "Download complete."