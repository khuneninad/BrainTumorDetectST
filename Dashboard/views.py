from django.shortcuts import render, redirect, get_object_or_404
from .forms import DataForm
from .models import Data

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = DataForm(request.POST, request.FILES)  # Handle image file uploads properly
        if form.is_valid():
            new_data = form.save()  # Save the form and get the newly created object
            return redirect('dashboard-result', pk=new_data.pk)  # Redirect to a result page
    else:
        form = DataForm()
    context = {
        'form': form,
    }
    return render(request, 'dashboard/index.html', context)

def predictions(request):
    predicted_tumor = Data.objects.all()  # Fetch all data entries
    context = {
        'predicted_tumor': predicted_tumor
    }
    return render(request, 'dashboard/predictions.html', context)

def result(request, pk):
    data_entry = get_object_or_404(Data, pk=pk)  # Fetch specific entry by primary key
    context = {
        'data_entry': data_entry
    }
    return render(request, 'dashboard/result.html', context)
