import time
import io
from flask import Flask, render_template, request,make_response,send_file
from zipfile import ZipFile
import os
import requests
import zipfile

app = Flask(__name__)
images = [
    {"id": 1, "url": '\static\image\image_1.jpg'},
    {"id": 2, "url": '\static\image\image_2.jpg'},
    {"id": 3, "url": '\static\image\image_3.jpg'},
]
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_text = request.form['user_text']
    #stat=request.form['option1'] #вернет : сюда_че_хочешь_вернуть
    time.sleep(2) #задержка для тестов
    return render_template("carousel.html", images=images,user_text=user_text)
@app.route("/download", methods=["POST"])


def download():
    selected_images_ids = request.form.getlist("selected_images")
    print(selected_images_ids)

    static_folder = r'C:\Users\Wdtum\PycharmProjects\gen_pict\static\image' # указываем полный путь до папки!!!!
    image_paths = []
    for image_id in selected_images_ids:
        image_filename = f'image_{image_id}.jpg'
        image_path = os.path.join(static_folder, image_filename)
        if os.path.exists(image_path):
            image_paths.append(image_path)
    zip_filename = 'images.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for image_path in image_paths:
            zipf.write(image_path, os.path.basename(image_path))

    # отправка архива пользователю
    return send_file(zip_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)