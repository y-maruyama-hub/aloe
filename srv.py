import os
import numpy as np
import cv2
import argparse
from flask import Flask,Response,request,jsonify,session
from dotenv import load_dotenv
import json
#import urllib.parse
import urllib.request
import base64
import time
import datetime
from datetime import timedelta
import traceback

import mitsuba.imcut as imcut


predicturl = None
savepath = None
bg = None
diff_thr = None
cutsize = None
allsize = None

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.secret_key = "hogefuga"


def init_session() :

    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)
    session.modified = True


@app.route("/detect",methods=["POST"])
def detect():
    global bg

    response = None
    code=200

    try:
        #data = request.data.decode('utf-8')

        #j = json.loads(data)

        #img = base64.b64decode(j["img"])
        #bgreq = j["bg"]

        #temp = request.json["img"]
        rjson = request.get_json()

        img = np.frombuffer(base64.b64decode(rjson["img"]), dtype=np.uint8)
        bgreq = rjson["bg"]

        #print(img)

        reqframe = cv2.imdecode(img,flags=cv2.IMREAD_COLOR)

        if bg is None :
            bg = cv2.cvtColor(reqframe, cv2.COLOR_BGR2GRAY)
            res={"res":-1,"img":rjson["img"]}
            response = jsonify(res)

        else :
            detec,frame = framediff(reqframe)

            _,jpeg = cv2.imencode('.jpg', frame)

            #imgstr = jpeg.tostring()
            imgstr = base64.b64encode(jpeg.tostring()).decode("utf-8")

            res={"res":detec,"img":imgstr}

            response = jsonify(res)


            if bgreq :
                bg = cv2.cvtColor(reqframe, cv2.COLOR_BGR2GRAY)

            #response.headers[""] = ""

    except :
        traceback.print_exc()
        code=500
    finally :
        return response,code


'''
@app.route("/detect",methods=["POST"])
def detect():

    response = None

    try:
        if "islogin" not in session : raise Exception

        _bytes = np.frombuffer(request.data, np.uint8)

        frame = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)

        detec,frame = framediff(frame)

        _,jpeg = cv2.imencode('.jpg', frame)

        #imgstr = jpeg.tostring()
        imgstr = base64.b64encode(jpeg.tostring()).decode("utf-8")

        res={"res":detec,"img":imgstr}

        response = jsonify(res)
        code=200
        #response.headers[""] = ""

    except :
        code=500
    finally :
        return response,code
#    return Response(jpeg.tobytes(),mimetype="image/jpeg")

'''


@app.route("/bgrenew",methods=["POST"])
def bgrenew():
    global bg

    if "islogin" not in session :
        init_session()
        session["islogin"]="1"

    _bytes = np.frombuffer(request.data, np.uint8)

    frame = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)

    bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res={"res":True}

    return jsonify(res)



def framediff(frame):
#    global bg

    if bg is None :
        return None

    writeImg("1",frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(bg,gray)

    _,frame_diff = cv2.threshold(frame_diff,50,255, cv2.THRESH_BINARY)
    frame_diff = cv2.medianBlur(frame_diff, 5)

    diff_point=cv2.countNonZero(frame_diff)

    det1 = 0
    det2 = 0

    retframe=frame.copy()

    if diff_thr < diff_point:
        contrs,hierarchy = cv2.findContours(frame_diff,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for pt in contrs:
            mu = cv2.moments(pt)

            if(mu["m00"]>2000):
                cx=int(mu["m10"]/mu["m00"])
                cy=int(mu["m01"]/mu["m00"])

                cutwx,cutwy = imcut.adjust_size(mu["m00"],0.6,cutsize)

                xx = imcut.cut_over(cx,cutwx,allsize[0])
                yy = imcut.cut_over(cy,cutwy,allsize[1])

                expimg=frame[yy[0]:yy[1],xx[0]:xx[1]]
                #expimg = cv2.resize(expimg,cutsize)

                color=(0, 255, 0)

                if predict(expimg)>0.7 :
                    writeImg("1",expimg)
                    det1 += 1
                    color=(255, 0, 0)
                    
                else :
                    writeImg("0",expimg)
                    det2 += 1

                cv2.circle(retframe, (cx,cy), 4, color, 2)
                cv2.rectangle(retframe,(xx[0],yy[0]),(xx[1],yy[1]),color, 1)

    detec = -1

    if det1>0 : detec = 1
    elif det2>0 : detec = 0

    return detec,retframe


def writeImg(dir,frame):
    tm = datetime.datetime.fromtimestamp(time.time())
    cv2.imwrite("{0}/{1}/{2}.jpg".format(savepath,dir,tm.strftime("%Y%m%d%H%M%S")),frame)



def predict(frame):

    #frame = cv2.resize(frame,(64,48))
    _,jpeg= cv2.imencode(".jpg", frame)

    req = urllib.request.Request(
        predicturl,
        jpeg.tobytes(),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )

    response = urllib.request.urlopen(req)
    json_str = response.read()
    response.close()

    j = json.loads(json_str)

    return j["prob"]




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-p","--port",type=int,default=5000)

    args = parser.parse_args()

    myport=int(args.port)

    load_dotenv()

    predicturl=os.getenv("PREDICTURL")
    savepath=os.getenv("SAVEPATH")

    diff_thr = int(os.getenv("DIFFTHR"))

    cutx = int(os.getenv("CUTX"))
    allx = int(os.getenv("ALLX"))
    ratioy = 0.75

    cutsize=(cutx,cutx*ratioy)
    allsize=(allx,allx*ratioy)

    app.run(host='0.0.0.0', debug=False,threaded=True,port=myport)
