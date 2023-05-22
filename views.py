from flask import Blueprint, render_template,request, redirect, url_for
from ner import loadModel, namedEntityRecogniser
import POSTagger
from sentimentanalyser import loadmodel, sentimentAnalyser

views = Blueprint(__name__, "views")

@views.route("/nerresult")
def nerresult() :
    return render_template("ner.html")

@views.route("/posresult")
def posresult() :
    return render_template("pos.html")

@views.route("/pos", methods = ['GET','POST'])
def pos() :

    if request.method == 'POST':
        nlp = POSTagger.loadModel()
        user_input = request.form.get("userinput")
        POSTagger.POSTagger(nlp=nlp, user_input=user_input)
        return redirect(url_for("views.posresult"))

    return render_template("posinput.html")

@views.route("/ner", methods = ['GET','POST'])
def ner() :

    if request.method == 'POST':
        nlp = loadModel()
        user_input = request.form.get("userinput")
        namedEntityRecogniser(nlp=nlp, user_input=user_input)
        return redirect(url_for('views.nerresult'))

    return render_template("nerinput.html")
@views.route("/", methods = ['GET','POST'])
def sentimentanalyser() :
    result=""
    input=""
    pos=""
    neg =""
    if request.method == 'POST' :
        classifier, vectorizer = loadmodel()
        user_input = request.form.get("userinput")
        input = '"'+user_input+'"'
        prediction = sentimentAnalyser(classifier=classifier, vectorizer=vectorizer, user_input=user_input)
        # prediction = classifier.predict_proba(user_input)
        if prediction[0][0] > prediction[0][1]:
            result = "Likely to be Negative\n"
            pos = "P = %0.2f" % (prediction[0][1]*100) + "%"
            neg = "N = %0.2f" % (prediction[0][0]*100) + "%\n"
        else:
            result = "Likely to be Positive\n"
            pos = "P= %0.2f" % (prediction[0][1]*100) + "%"
            neg = "N= %0.2f" % (prediction[0][0]*100) + "%\n"

    return render_template("home.html",pos= pos, neg=neg,input = input,result= result)