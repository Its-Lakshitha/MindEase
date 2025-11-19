from django.shortcuts import render
from django.utils import timezone

def landing(request):
    # optional context
    ctx = {"now": timezone.now()}
    return render(request, "landing.html", ctx)

def aboutus(request):
    ctx = {"now": timezone.now()}
    return render(request, "aboutus.html", ctx)

def contactus(request):
    ctx = {"now": timezone.now()}
    if request.method == "POST":
        # ...existing code...
        # here you would process the form (send email, save to DB, etc.)
        ctx["message_sent"] = True
    return render(request, "contactus.html", ctx)

