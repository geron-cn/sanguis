from django.db import models
from django.contrib import admin
from rest_framework import serializers
from django.http import HttpResponse, HttpResponseRedirect,Http404
import django.utils.timezone as timezone
from daterange_filter.filter import DateRangeFilter
from .TFModel import TFModel
from django_admin_listfilter_dropdown.filters import (
	DropdownFilter, ChoiceDropdownFilter, RelatedDropdownFilter
)


class Data(models.Model):
	'''预测数据'''
	data        = models.CharField('预测数据',max_length=400, unique=False, blank = False)
	predict     = models.FloatField('预测结果', unique=False, blank = False)
	expect      = models.FloatField('期望结果', unique=False, blank = True)
	model     = models.ForeignKey(TFModel, to_field='name', on_delete=models.CASCADE, default=None, verbose_name='模型')
	create_time = models.DateTimeField('创建时间', auto_now=True)
	isTF        = models.BooleanField('是否训练数据', default=False)

	class Meta:
		ordering = ['-create_time']
		verbose_name = '预测数据'

	def __str__(self):
		return self.model.name + '_' +str(self.id)

	def dataArr(self):
		farr = [float(x) for x in self.data.split(',')]
		return farr

class DataAdmin(admin.ModelAdmin):
	list_display = ('model', 'data','predict','expect', 'isTF','create_time',)
	list_filter = (
		('model', RelatedDropdownFilter), 
		('isTF', DropdownFilter),
		('create_time', DateRangeFilter)
	)

	search_fields = ['model__name']

class DataSerializer(serializers.ModelSerializer):
	class Meta:
		model = Data
		fields = ['data','predict', 'expect', 'model', 'isTF','create_time']




