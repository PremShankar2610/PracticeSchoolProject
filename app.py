from flask import Flask, render_template, request,Response,jsonify
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
#from flask_ngrok import run_with_ngrok
from binascii import a2b_base64
from PIL import Image
import io,base64
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)
#run_with_ngrok(app)
class_car_defect=['With defect','Not defective']
class_for_fashion = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle boot']
class_for_intel=['buildings','forest','glacier','mountain','sea','street']
class_for_oxford=["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			   "windflower", "pansy"]
dic = {i:j for i,j in enumerate(class_for_fashion)}
image_shape_for_fashion=(28,28)
image_shape_for_intel=(150,150)
image_shape_for_oxford=(224,224)
image_shape_for_car_defect=(224,224)
model = load_model('car_defect_detector.model')

#model.make_predict_function()

def predict_label(img_path):
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,image_shape_for_car_defect)
    print(img.shape)
    images=np.array(img,dtype='float32')
    image = preprocess_input(images)
    print(image.shape)
    image=np.expand_dims(image, axis=0)
    pred=model.predict(image)
    return(class_car_defect[int(np.argmax(pred,axis=1))])


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

@app.route('/submit_out',methods=['GET','POST'])
def camera_rear():
    if(request.method=='POST'):
        img=request.files['image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
    return render_template("index.html", prediction = p, img_path = img_path)

        
@app.route('/submit_camera', methods=['GET','POST'])
def camera_front():
    if request.method == 'POST':
        img=request.form['image']
        im = Image.open(io.BytesIO(base64.b64decode(img.split(',')[1])))
        im.save(r'static\prem.jpg', 'JPEG')
        #cv2.imwrite("test.jpg", img)
        img_path=r'static\prem.jpg'
        p=predict_label(img_path)
    return (str(p)) 


if __name__ =='__main__':
	#app.debug = True
	app.run(host="localhost", port=8000, debug=True)
