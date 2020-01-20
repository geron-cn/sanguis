"""sanguis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from .models.Data import *
from .models.TFModel import *
from .views import apiview

admin.site.site_header = '血液AI智能管理系统'
admin.site.site_title = '血液AI智能管理系统'

admin.site.register(Data, DataAdmin)
admin.site.register(TFModel, TFModelAdmin)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('currentmodel/', apiview.currentModel),
    path('listmodeldatas/', apiview.listModels),
    path('trainmodel/<str:name>/',apiview.trainModel),
    re_path('^predict/$', apiview.predict)
]
