from flask import Flask, render_template, request, url_for, send_file, Response, render_template_string
from werkzeug.utils import secure_filename
import os
from textblob import TextBlob
import cv2
from googletrans import Translator
from gtts import gTTS

from generate_caption import get_vgg16_caption, get_inceptionV3_caption, get_resnet_caption
from semantic_search import get_similars


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "ui_app/static"



@app.route("/")
@app.route("/index")
@app.route("/index/")
def index():
    return render_template("index.html")

@app.route('/camera')
@app.route('/camera/')
def video_feed():
    #return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template_string('''
<video id="video" width="640" height="480" autoplay style="background-color: grey"></video>
<button id="send">Take & Send Photo</button>
<canvas id="canvas" width="640" height="480" style="background-color: grey"></canvas>

<script>

// Elements for taking the snapshot
var video = document.getElementById('video');
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
    });
}

// Trigger photo take
document.getElementById("send").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480); // copy frame from <video>
    canvas.toBlob(upload, "image/jpeg");  // convert to file and execute function `upload`
});

function upload(file) {
    // create form and append file
    var formdata =  new FormData();
    formdata.append("snap", file);
    
    // create AJAX requests POST with file
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "{{ url_for('upload') }}", true);
    xhr.onload = function() {
        if(this.status = 200) {
            console.log(this.response);
        } else {
            console.error(xhr);
        }
        window.location.replace("webcam_load")
        //alert(this.response);
    };

    xhr.send(formdata);
}

</script>
''')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #fs = request.files['snap'] # it raise error when there is no `snap` in form
        fs = request.files.get('snap')
        if fs:
            #print('FileStorage:', fs)
            print('filename:', fs.filename)
            path = app.config['UPLOAD_FOLDER']+"/"+'image_webcam.jpg'
            fs.save(path)
            return 'Snap!'
        else:
            return 'You forgot Snap!'			

@app.route('/camera/webcam_load', methods=['GET', 'POST'])
@app.route('/camera/webcam_load/', methods=['GET', 'POST'])
def webcam_load():
		path = 'ui_app/static/image_webcam.jpg'
		filename = 'image_webcam.jpg'
		#caption = get_vgg16_caption(path)
		#caption = get_inceptionV3_caption(path)
		caption_text = get_resnet_caption(path)
		translator = Translator()
		translated = translator.translate(caption_text, dest = 'fr')
		caption_text = translated.text
		audio = gTTS(caption_text, lang='fr')
		audio.save("ui_app/static/audio.mp3")
		return render_template("camera.html", filename=filename, caption_text=caption_text)



@app.route("/generator", methods=["GET", "POST"])
@app.route("/generator/", methods=["GET", "POST"])
def generator():
	if request.method == "GET":
		return render_template("generator.html")
	elif request.method == "POST":
		# save the image file, generate the caption and show it in hmtl page
		uploaded_file = request.files["image_file"]
		filename = secure_filename(uploaded_file.filename)
		path = app.config['UPLOAD_FOLDER']+"/"+filename
		uploaded_file.save(path)				
		vgg16_caption = get_vgg16_caption(path)
		inceptionV3_caption = get_inceptionV3_caption(path)
		resnet_caption = get_resnet_caption(path)
		return render_template("generator_load.html", filename=filename, vgg16_caption=vgg16_caption, inceptionV3_caption=inceptionV3_caption, resnet_caption=resnet_caption)


@app.route("/image_search", methods=["GET", "POST"])
@app.route("/image_search/", methods=["GET", "POST"])
def image_search():
	if request.method == "GET":
		return render_template("image_search.html")
	elif request.method == "POST":
		query = request.form["query"]
		similar_files, similar_captions = get_similars(query)
		return render_template("image_hits.html", query=query, similar_files=similar_files, similar_captions=similar_captions)


@app.route("/models")
@app.route("/models/")
def models():
    return render_template("models.html")

@app.route("/analysis", methods=["GET", "POST"])
@app.route("/analysis/", methods=["GET", "POST"])
def analysis():
	if request.method == 'POST':
		inp = request.form.get("text")
		edu = TextBlob(inp)
		x = edu.sentiment.polarity
		if x < 0:
			return render_template('analysis.html', message = "Negative 🙁")
		elif x==0:
			return render_template('analysis.html', message = "Neutral 😶")
		elif x>0 and x<=1:
			return render_template('analysis.html', message = "Positive 🙂")
    
	return render_template("analysis.html")
