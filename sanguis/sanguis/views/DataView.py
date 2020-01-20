from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework import serializers
from rest_framework.decorators import api_view
from django.shortcuts import render
from ..models.Data import Data
from ..models.TFModel import TFModel