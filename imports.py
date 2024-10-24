import numpy as np
import pandas as pd
import requests , json ,pytz,ccxt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split, cross_val_score
# from telegram import Bot
import telebot
import time
import matplotlib.pyplot as plt
import hmac
import hashlib
import ta
import os
import MetaTrader5 as mt5
from datetime import datetime ,timedelta
import threading
import asyncio
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, make_scorer,precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
from datetime import timezone
from scipy.stats import linregress
from sklearn.metrics import *
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Input,Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
import joblib
from scikeras.wrappers import KerasClassifier, KerasRegressor

import multiprocessing
import matplotlib.dates as mpl_dates 
import matplotlib.dates as mdates
# import BeautyChart
import multiprocessing
