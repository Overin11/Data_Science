#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Загрузка-и-подготовка-данных" data-toc-modified-id="Загрузка-и-подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Загрузка и подготовка данных</a></span></li><li><span><a href="#Обучение-и-проверка-модели" data-toc-modified-id="Обучение-и-проверка-модели-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Обучение и проверка модели</a></span><ul class="toc-item"><li><span><a href="#Разбил-данные-на-обучающую-и-валидационную-выборки-в-соотношении-75:25." data-toc-modified-id="Разбил-данные-на-обучающую-и-валидационную-выборки-в-соотношении-75:25.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Разбил данные на обучающую и валидационную выборки в соотношении 75:25.</a></span></li><li><span><a href="#Обучил-модель-и-сделал-предсказания-на-валидационной-выборке." data-toc-modified-id="Обучил-модель-и-сделал-предсказания-на-валидационной-выборке.-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Обучил модель и сделал предсказания на валидационной выборке.</a></span></li><li><span><a href="#Напечатайте-на-экране-средний-запас-предсказанного-сырья-и-RMSE-модели" data-toc-modified-id="Напечатайте-на-экране-средний-запас-предсказанного-сырья-и-RMSE-модели-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Напечатайте на экране средний запас предсказанного сырья и RMSE модели</a></span></li></ul></li><li><span><a href="#Подготовка-к-расчёту-прибыли" data-toc-modified-id="Подготовка-к-расчёту-прибыли-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Подготовка к расчёту прибыли</a></span><ul class="toc-item"><li><span><a href="#Все-ключевые-значения-для-расчётов-сохранил-в-отдельных-переменных" data-toc-modified-id="Все-ключевые-значения-для-расчётов-сохранил-в-отдельных-переменных-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Все ключевые значения для расчётов сохранил в отдельных переменных</a></span></li><li><span><a href="#Рассчитал-достаточный-объём-сырья-для-безубыточной-разработки-новой-скважины.-Сравнил-полученный-объём-сырья-со-средним-запасом-в-каждом-регионе." data-toc-modified-id="Рассчитал-достаточный-объём-сырья-для-безубыточной-разработки-новой-скважины.-Сравнил-полученный-объём-сырья-со-средним-запасом-в-каждом-регионе.-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Рассчитал достаточный объём сырья для безубыточной разработки новой скважины. Сравнил полученный объём сырья со средним запасом в каждом регионе.</a></span></li></ul></li><li><span><a href="#Расчёт-прибыли-и-рисков" data-toc-modified-id="Расчёт-прибыли-и-рисков-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Расчёт прибыли и рисков</a></span><ul class="toc-item"><li><span><a href="#Выбрал-скважины-с-максимальными-значениями-предсказаний,-просуммировал-целевое-значение-объёма-сырья,-соответствующее-этим-предсказаниям.-Рассчитал-прибыль-для-полученного-объёма-сырья." data-toc-modified-id="Выбрал-скважины-с-максимальными-значениями-предсказаний,-просуммировал-целевое-значение-объёма-сырья,-соответствующее-этим-предсказаниям.-Рассчитал-прибыль-для-полученного-объёма-сырья.-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Выбрал скважины с максимальными значениями предсказаний, просуммировал целевое значение объёма сырья, соответствующее этим предсказаниям. Рассчитал прибыль для полученного объёма сырья.</a></span></li><li><span><a href="#Графики" data-toc-modified-id="Графики-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Графики</a></span></li></ul></li><li><span><a href="#Рассчёт-риски-и-прибыль-для-каждого-региона:" data-toc-modified-id="Рассчёт-риски-и-прибыль-для-каждого-региона:-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Рассчёт риски и прибыль для каждого региона:</a></span><ul class="toc-item"><li><span><a href="#Применил-технику-Bootstrap-с-1000-выборок,-чтобы-найти-распределение-прибыли." data-toc-modified-id="Применил-технику-Bootstrap-с-1000-выборок,-чтобы-найти-распределение-прибыли.-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Применил технику Bootstrap с 1000 выборок, чтобы найти распределение прибыли.</a></span></li><li><span><a href="#Нашёл-среднюю-прибыль,-95%-й-доверительный-интервал-и-риск-убытков." data-toc-modified-id="Нашёл-среднюю-прибыль,-95%-й-доверительный-интервал-и-риск-убытков.-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Нашёл среднюю прибыль, 95%-й доверительный интервал и риск убытков.</a></span></li></ul></li><li><span><a href="#Вывод" data-toc-modified-id="Вывод-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Вывод</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Выбор локации для скважины

# Допустим, вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.
# 
# Вам предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой *Bootstrap.*
# 
# Шаги для выбора локации:
# 
# - В избранном регионе ищут месторождения, для каждого определяют значения признаков;
# - Строят модель и оценивают объём запасов;
# - Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
# - Прибыль равна суммарной прибыли отобранных месторождений.

# -    id — уникальный идентификатор скважины;
# -    f0, f1, f2 — три признака точек (неважно, что они означают, но сами признаки значимы);
# -    product — объём запасов в скважине (тыс. баррелей).

# ## Загрузка и подготовка данных

#     Импортирую все необходимые библиотеки в начале проекта

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st


#     Загружаю данные по трем регионам

# In[2]:


data_0 = pd.read_csv('/datasets/geo_data_0.csv')
data_1 = pd.read_csv('/datasets/geo_data_1.csv')
data_2 = pd.read_csv('/datasets/geo_data_2.csv')


#     Исследую данные

# In[3]:


data_0.info()
data_1.info()
data_2.info()


# Нулевых значений нет

# In[4]:


display(data_0['product'].describe(), 
      data_1['product'].describe(), 
      data_2['product'].describe(), 
      #sep='\n'
      )


#     В данных, в столбце 'product' есть 0, т.е. это объём запасов в скважине (тыс. баррелей), оставляю эти данные бесполезны, т.к. они пригодятся при исследовании.
#     Далее проверяю наличие повторов среди id — уникальных идентификаторов скважин

# In[5]:


display(
    data_0['id'].duplicated().sum(), 
    data_1['id'].duplicated().sum(), 
    data_2['id'].duplicated().sum(), 
    )


#     Не очень уникальные, удаляю повторы

# In[6]:


data_0 = data_0.drop_duplicates(['id'])
data_1 = data_1.drop_duplicates(['id'])
data_2 = data_2.drop_duplicates(['id'])


#     Т.к. для обучения модели столбец 'id' не понадобится, удаляю

# In[7]:


data_0 = data_0.drop(['id'], axis = 1)
data_1 = data_1.drop(['id'], axis = 1)
data_2 = data_2.drop(['id'], axis = 1)


# ## Обучение и проверка модели

# ### Разбил данные на обучающую и валидационную выборки в соотношении 75:25.

# In[8]:


X_0 = data_0.drop(['product'], axis = 1)
y_0 = data_0['product']

X_1 = data_1.drop(['product'], axis = 1)
y_1 = data_1['product']

X_2 = data_2.drop(['product'], axis = 1)
y_2 = data_2['product']


# In[9]:


X_train_0, X_valid_0, y_train_0, y_valid_0 = train_test_split(
    X_0, y_0, test_size=0.25, random_state=1234)

X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
    X_1, y_1, test_size=0.25, random_state=1234)

X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(
    X_2, y_2, test_size=0.25, random_state=1234)


# ### Обучил модель и сделал предсказания на валидационной выборке.

# In[10]:


lr_0 = LinearRegression()
lr_0.fit(X_train_0, y_train_0)

lr_1 = LinearRegression()
lr_1.fit(X_train_1, y_train_1)

lr_2 = LinearRegression()
lr_2.fit(X_train_2, y_train_2)


# In[11]:


predictions_0 = lr_0.predict(X_valid_0) 
predictions_1 = lr_1.predict(X_valid_1)
predictions_2 = lr_2.predict(X_valid_2)


# ### Напечатайте на экране средний запас предсказанного сырья и RMSE модели

# In[12]:


#mse_0 = mean_squared_error(target_valid_0, predictions_0)
#rmse_0 = mse_0 ** 0.5 

#mse_1 = mean_squared_error(target_valid_1, predictions_1)
#rmse_1 = mse_1 ** 0.5 

#mse_2 = mean_squared_error(target_valid_2, predictions_2)
#rmse_2 = mse_2 ** 0.5 


# In[13]:


#display("RMSE #1: {: >20.2f} тыс. баррелей".format(rmse_0))
#display('Предсказания {: >16.2f} тыс. баррелей'.format(np.mean(predictions_0)))
#display("RMSE #2: {: >19.2f} тыс. баррелей".format(rmse_1))
#display('Предсказания {: >16.2f} тыс. баррелей'.format(np.mean(predictions_1)))
#display("RMSE #3: {: >20.2f} тыс. баррелей".format(rmse_2))
#display('Предсказания {: >16.2f} тыс. баррелей'.format(np.mean(predictions_2)))


# In[14]:


y_valid = [y_valid_0, y_valid_1, y_valid_2] 
predictions = [predictions_0, predictions_1, predictions_2]
count = 1
RMSE_list = []
predict_list = []
def mse_rmse(y_valid, predictions):
    mse = mean_squared_error(y_valid, predictions)
    rmse = mse ** 0.5 
    RMSE_list.append(rmse.round(2))
    predict_list.append(np.mean(predictions).round(2))
    print("RMSE модели {} региона: {: >20.2f} тыс. баррелей".format(count, rmse))
    print('Предсказания {} региона {: >20.2f} тыс. баррелей'.format(count, np.mean(predictions)))


for i, j in zip(y_valid, predictions):
    mse_rmse(i, j)
    count += 1


#     Создам таблицу для записи результатов 

# In[15]:


resultation_table = pd.DataFrame(
    {'Регионы' : ['Первый', 'Второй', 'Третий'],
     'RMSE модели региона' : RMSE_list, 
     'Предсказания ' : predict_list
     }
)


# In[16]:


corr_0 = data_0.corr(method="pearson")
corr_1 = data_1.corr(method="pearson")
corr_2 = data_2.corr(method="pearson")
display('Превый регион', corr_0,
        'Второй регион', corr_1, 
        'Третий регион', corr_2, 
        )


# In[17]:


for plot in [data_0, data_1, data_2]:
    plt.figure(figsize=(12,10), dpi= 80)
    sns.heatmap(plot.corr(), 
                #xticklabels=data_0.corr().columns, 
                #yticklabels=data_0.corr().columns, 
                cmap='coolwarm', # посмотреть другие цвета 
                center=0, # ставит центр цветового ряда на 0 
                annot=True # аннотация(цифры на графике)
                )

    plt.title('Корреляция данных', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


#     Вывод: если судить по предсказаниям модели, то самым перспективыным является третий регион с запасом нефти объемом 94.89 тысяч баррелей, затем идет первый регион с 92.31 тысячами баррелей и потом второй регион с 75.16 тысячами баррелей.
#     Но если взять в расчет среднеквадратичную ошибку, то она самая большая у предсказаний третьего региона, самая маленькая - у первого, меньше 1, из-за сильной корреляции 3 признака и целевого параметра

# ## Подготовка к расчёту прибыли

# ### Все ключевые значения для расчётов сохранил в отдельных переменных

# In[18]:


# Бюджет на разработку месторождений
BUDGET = 10_000_000_000
WELL_COST = 50_000_000
well_number = BUDGET // WELL_COST
# доход с каждой единицы продукта 1 тыс. баррелей
INCOME = 450_000


# ### Рассчитал достаточный объём сырья для безубыточной разработки новой скважины. Сравнил полученный объём сырья со средним запасом в каждом регионе.

# In[19]:


volume = BUDGET / INCOME
print(
    'Достаточный объём сырья для безубыточной разработки новой скважины: {: >12.2f} тыс. баррелей'.format(volume), 
    'Средний запас в первом регионе: {: >50.2f} тыс. баррелей'.format(np.mean(sum(data_0['product']))), 
    'Одна скважина в первом регионе в среднем принесет{: >28.2f} млрд.'.format(np.mean(sum(data_0['product'])) / 200 * INCOME / 1000000000), 
    'Средний запас во втором регионе: {: >49.2f} тыс. баррелей'.format(np.mean(sum(data_1['product']))), 
    'Одна скважина в первом регионе в среднем принесет{: >28.2f} млрд.'.format(np.mean(sum(data_1['product'])) / 200 * INCOME / 1000000000), 
    'Средний запас в третьем регионе: {: >49.2f} тыс. баррелей'.format(np.mean(sum(data_2['product']))), 
    'Одна скважина в первом регионе в среднем принесет{: >28.2f} млрд.'.format(np.mean(sum(data_2['product'])) / 200 * INCOME / 1000000000),
    sep='\n'
    )
mean_vol_list = [
    np.mean(sum(data_0['product'])).round(2), 
    np.mean(sum(data_1['product'])).round(2), 
    np.mean(sum(data_2['product'])).round(2)
]
mean_well = [
    (np.mean(sum(data_0['product'])) / 200 * INCOME / 1000000000).round(2), 
    (np.mean(sum(data_1['product'])) / 200 * INCOME / 1000000000).round(2), 
    (np.mean(sum(data_2['product'])) / 200 * INCOME / 1000000000).round(2)
    ]


#     Вывод: при вложении 10 млрд. рублей, для безубыточной разработки, запасы должны быть не менее 22222.22 тыс. баррелей. Все 3 региона соответствуют этому критерию, средний запас больше этого значения

# In[20]:


resultation_table['Средний запас в регионе (тыс. баррелей)'] = mean_vol_list
resultation_table['Одна скважина в регионе в среднем принесет (млрд.)'] = mean_well


# ## Расчёт прибыли и рисков 

# ### Выбрал скважины с максимальными значениями предсказаний, просуммировал целевое значение объёма сырья, соответствующее этим предсказаниям. Рассчитал прибыль для полученного объёма сырья.

# In[21]:


def calculate(data, features, lr):
    data['predict'] = lr.predict(features)
    data_income = data.sort_values('predict', ascending=False).head(500)
    raw_volume = sum(data_income['product'])        
    income_volume = sum(data_income['product']) * 450 / 1000000
    target_value_list.append(round(raw_volume, 2))
    income_list.append(round(income_volume, 2))
    print('Сумма целевого значения сырья: {} региона {: >11.2f} тыс баррелей'.format(count, raw_volume))
    print('Прибыль {} региона: {: >30.2f} млрд рублей'.format(count, income_volume))
    print()
target_value_list = []
income_list = []
data = [data_0, data_1, data_2]
features = [X_0, X_1, X_2]
lr = [lr_0, lr_1, lr_2]
count = 1
for  i, j, l in zip (data, features, lr):
    calculate(i, j, l)
    count += 1


# In[22]:


resultation_table['Сумма целевого значения сырья (тыс. баррелей)'] = target_value_list
resultation_table['Прибыль региона (млрд. рублей)'] = income_list


# ### Графики

# In[23]:


data_income_0 = data_0.sort_values('predict', ascending=False).head(500)
data_income_1 = data_1.sort_values('predict', ascending=False).head(500)
data_income_2 = data_2.sort_values('predict', ascending=False).head(500)

income_volume_0 = sum(data_income_0['product']) * 450 / 1000000
income_volume_1 = sum(data_income_1['product']) * 450 / 1000000
income_volume_2 = sum(data_income_2['product']) * 450 / 1000000
    
data_income = pd.DataFrame({'Суммарный объем первого региона' : [sum(data_income_0['product'])], 
                            'Суммарный объем второго региона' : [sum(data_income_1['product'])], 
                            'Суммарный объем третьего региона' : [sum(data_income_2['product'])], }
                            )
data_income.plot(kind ='bar', 
                 grid = True, 
                 alpha = 0.6, 
                 )
plt.xlabel('Регионы')
plt.ylabel('тыс баррелей')
plt.legend(loc='lower left', 
           bbox_to_anchor=(1, 0.5)
           )
plt.title('Суммарный объем сырья', fontsize=22)
plt.xticks(fontsize=1)
plt.yticks(fontsize=12)
plt.show()    
    
income_volume = pd.DataFrame({'Прибыль первого региона' : [income_volume_0], 
                            'Прибыль второго региона' : [income_volume_1], 
                            'Прибыль третьего региона' : [income_volume_2]}
                            )
income_volume.plot(kind ='bar', 
                    grid = True, 
                    alpha = 0.6, 
                    
                    )
plt.ylabel('млрд рублей')
plt.xlabel('Регионы')
plt.legend(loc='lower left',
            bbox_to_anchor=(1, 0.5))
plt.title('Прибыль регионов', fontsize=22)
plt.xticks(fontsize=1)
plt.yticks(fontsize=12)
plt.show()


# ## Рассчёт риски и прибыль для каждого региона:

# ### Применил технику Bootstrap с 1000 выборок, чтобы найти распределение прибыли.

# ### Нашёл среднюю прибыль, 95%-й доверительный интервал и риск убытков.

# In[24]:


# BUDGET = 10000000000
# WELL_NUMBER = 200
# WELL_COST = BUDGET / WELL_NUMBER
# INCOME = 450000

def profit(y, predictions):
    predictions_sorted = predictions.sort_values(ascending=False)
    selected_points = y[predictions_sorted.index][:well_number]
    product = selected_points.sum()
    revenue = product * INCOME
    cost = WELL_COST * well_number
    return revenue - cost

state = np.random.RandomState(1234)
    
values = []

data_0['predictions'] = lr_0.predict(X_0) 
data_1['predictions'] = lr_1.predict(X_1) 
data_2['predictions'] = lr_2.predict(X_2) 

mean_plot = [] #plot для графиков средней выручки

y_valid_0
data_0['predictions']

def risk_income(y_valid, data):
    values = []
    for i in range(1000):
        target_subsample_0 = y_valid.sample(n=500, replace=True, random_state=state)
        probs_subsample_0 = data[target_subsample_0.index]
        values.append(profit(target_subsample_0, probs_subsample_0))

    values = pd.Series(values)
    lower = values.quantile(0.05) 
    mean = values.mean()
    mean_income_list.append(round(mean / 1000000, 2))
    #print(mean)
    mean_plot.append(mean)
    confidence_interval = st.t.interval(0.95, len(values) - 1, values.mean(), values.sem()) # < напишите код здесь >
    #loss = (values < 0).mean()
    loss = st.percentileofscore(values, 0) #показывает % в values меньше второго аргумента
    loss_list.append(round(loss, 2))
    
    print("Средняя выручка {} региона: {: >20.2f} млрд. рублей".format(count, mean / 1000000))
    #print("95%-ый доверительный интервал {} региона: {}".format(count, confidence_interval))
    print("95%-ый доверительный интервал {} региона: {}".format(count, [np.quantile(values, 0.025), np.quantile(values, 0.975)]))
    print("Убыток {} региона: {: >27.2f}".format(count, loss))
    print()

mean_income_list =[]
loss_list = []
count = 1   
#target_valid = [target_valid_0, target_valid_1, target_valid_2]    
data_012 = [data_0['predictions'], data_1['predictions'], data_2['predictions']]
for  i, j in zip (y_valid, data_012):
    risk_income(i, j)
    count += 1


# In[25]:


resultation_table['Средняя выручка региона (млрд. рублей)'] = mean_income_list
resultation_table['Убыток региона'] = loss_list


# In[26]:


# Вывод до исправления

# Средняя выручка 1 региона:               428.76 млрд. рублей
# 95%-ый доверительный интервал 1 региона: (411837723.74485, 445678271.9971178)
# Убыток 1 региона:                        0.06
# 
# Средняя выручка 2 региона:               487.94 млрд. рублей
# 95%-ый доверительный интервал 2 региона: (474936723.87195045, 500936182.20507175)
# Убыток 2 региона:                        0.01
# 
# Средняя выручка 3 региона:               397.83 млрд. рублей
# 95%-ый доверительный интервал 3 региона: (380299360.1455248, 415350658.0864517)
# Убыток 3 региона:                        0.09


# In[27]:


mean_plot = pd.DataFrame({'Выручка первого региона' : [mean_plot[0]], 
                          'Выручка второго региона' : [mean_plot[1]], 
                          'Выручка третьего региона' : [mean_plot[2]]}
                         )


# In[28]:


mean_plot.plot(kind ='bar', 
               grid = True, 
               alpha = 0.6, 
               legend = True
              )
plt.ylabel('млрд рублей')
plt.xlabel('Регионы')
plt.legend(loc='lower left',
           bbox_to_anchor=(1, 0.5)
           )
plt.title('Выручка регионов', fontsize=22)
plt.xticks(fontsize=1)
plt.yticks(fontsize=12)
plt.show()


# ## Вывод

# In[29]:


display(resultation_table)


#     Вывод: Во время исследования я использовал модель линейной регрессии для опредения региона который принесет наибольшую прибыль. После разделения данных на обучающую и тестовую выборки и обучение модели, я получил такие результаты предсказаний запасов объема нефти: 
#         Первый регион - 92.42 тыс. баррелей,
#         Второй регион - 74.94 тыс. баррелей,
#         Третий регион - 94.94 тыс. баррелей.
#     Среднеквадратичная ошибка: 
#         Первый регион - 37.62 тыс. баррелей,
#         Второй регион - 0.89 тыс. баррелей,
#         Третий регион - 40.22 тыс. баррелей.
#     Предсказания запасов делают третий регион самым привлекательным для инвестиций, но среднеквадратичная ошибка указывает на то, что нужно провести дополнительные исследования.
#     
#     Рассчитал достаточный объем сырья для безубыточной разработки. Все регионы соответствуют этому параметру.
#     Затем выбрал 500 скважин с максимальными значениями предсказаний. Просуммировал целевое значение объёма сырья, соответствующее этим предсказаниям. Рассчитал прибыль для полученного объёма сырья.
#     Прибыль 1 региона: 33.64 млрд рублей
#     Прибыль 2 региона: 31.04 млрд рублей
#     Прибыль 3 региона: 31.65 млрд рублей
#     Разница в почти 2 млрд, при вложении 100 млрд. Хотя самым прибыльным остался 1 регион.
#     После этого применил технику Bootstrap с 1000 выборок, чтобы найти распределение прибыли. Нашёл среднюю прибыль, 95%-й доверительный интервал и риск убытков. 
#     В результате получил такие результаты:
#     Средняя выручка 1 региона:    436.65 млрд. рублей
#     Убыток 1 региона:             0.05
# 
#     Средняя выручка 2 региона:    768.25 млрд. рублей
#     Убыток 2 региона:             0.00
# 
#     Средняя выручка 3 региона:    433.73 млрд. рублей
#     Убыток 3 региона:             0.07
#     
#     При расчёте модели у второго региона оказалось самое низкое предсказание запаса объема нефти. При этом самая маленькая среднеквадратичная ошибка. Прибыль от разработки скважин 2 региона уступает остальным, но разница 2 млрд при 100 млрд инвестиций. 
#     После применения техники bootstrap средняя выручка оказалась самой большой, поэтому предлагаю именно 2 регион для разработки скважин.
