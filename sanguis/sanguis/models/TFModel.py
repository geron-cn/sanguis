import os
from django.db import models
from django.contrib import admin
from rest_framework import serializers
from django.http import HttpResponse, HttpResponseRedirect,Http404
from daterange_filter.filter import DateRangeFilter
from django.conf import settings
from django_admin_listfilter_dropdown.filters import (
	DropdownFilter, ChoiceDropdownFilter, RelatedDropdownFilter
)

def uploadto(instance, filename):
	return settings.MEDIA_ROOT.join([instance.name, '_', filename])

class TFModel(models.Model):
	'''模型训练数据'''

	def __str__(self):
		return self.name

	name        = models.CharField('模型名称',max_length=200, unique=True, blank = False)
	datapath    = models.FileField('模型数据xlsx文件',upload_to=uploadto, max_length=200)
	create_time = models.DateTimeField('创建时间', auto_now=True)
	# data        = models.ForeignKey(Data, on_delete=models.CASCADE, default=None, verbose_name='预测数据')
	rmse        = models.FloatField('均方根误差', unique=False, default=0.0, blank = False)
	mae         = models.FloatField('平均绝对误差', unique=False, default=0.0, blank = False)
	

	class Meta:
		ordering = ['-create_time']
		verbose_name = '模型训练数据'

class TFModelAdmin(admin.ModelAdmin):
	list_display = ('name', 'rmse', 'mae', 'create_time')

	list_filter = (
		('name', DropdownFilter), 
		('datapath', DropdownFilter),
		('create_time', DateRangeFilter)
	)

	search_fields = ['name']

class TFModelSerializer(serializers.ModelSerializer):
	class Meta:
		model = TFModel
		fields = ['name','datapath', 'create_time', 'rmse', 'mae']