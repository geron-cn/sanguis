from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework import serializers
from rest_framework.decorators import api_view
from django.shortcuts import render
from ..models.Data import Data, DataSerializer
from ..models.TFModel import *
from ..tools import BPNN_YG

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


currModel = None


@api_view(['GET'])
def currentModel(request):
	if request.method == 'GET':
		if currModel == None:
			errmsg = 'none model, please use "trainmodel" create a model'
			logger.error(errmsg)
			return Response(errmsg)
		serializer = TFModelSerializer(currModel)
		return Response(serializer.data)

@api_view(['GET'])
def listModels(request):
	if request.method == 'GET':
		tfms = TFModel.objects.all()
		serializer = TFModelSerializer(tfms, many=True)
		logger.info('current model is %s'%serializer.data)
		return Response(serializer.data)
	return Response('Bad request')

@api_view(['GET'])
def trainModel(request, name):
	if request.method == 'GET':
		model = TFModel.objects.get(name=name)
		logger.info(model)
		if model == None:
			errmsg = 'no model named %s, please contact admin upload this model data'% name
			logger.error(errmsg)
			return Response(errmsg)
		modelpath = model.datapath
		rmse, mae = BPNN_YG.trainModel(modelpath)
		model.rmse = rmse
		model.mae = mae
		model.save()
		global currModel
		currModel = model
		serializer = TFModelSerializer(model)
		return Response(serializer.data)
	return Response('Bad request or train unkonw error')

@api_view(['POST'])
def predict(request):
	global currModel
	if currModel == None:
			errmsg = 'none model, please use "trainmodel" create a model'
			logger.error(errmsg)
			return Response(errmsg)

	if request.method == 'POST':
		logger.info(request.data)
		if not 'data' in request.data:
			return Response('bad request, not json or no "data" key')
		data = Data()
		data.data = request.data['data']
		data.model = currModel
		data.isTF = False
		dataarr = data.dataArr()
		data.predict = BPNN_YG.predict(dataarr)
		data.expect = 0.0
		data.save()
		serializer = DataSerializer(data)
		return Response(serializer.data)
	return Response('predict unkonw error')



