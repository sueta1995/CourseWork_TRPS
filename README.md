# Курсовая работа по ТРПС, 5 семестр

<p align="left">
  <img alt="Static Badge" src="https://img.shields.io/badge/python-3.10.12-purple">
  <img alt="Static Badge" src="https://img.shields.io/badge/tensorflow-2.10.0-green">
  <img alt="Static Badge" src="https://img.shields.io/badge/opencv-4.8.0.74-red">
  <img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-orange">
</p>

В данном проекте реализуется веб-приложение, которое создает карту мест обитания различных видов стрекоз Москвы. Пользователь регистрируется или заходит в аккаунт, присылает фотографию стрекозы и указывает местоположение находки, а обученная модель предсказывает вид, автоматически на карте создается метка, в которой содержится фотография и информация о предсказанном виде. Если меток с одним видом становится достаточно большим в одной зоне, то они объединяются. Пользователь может просматривать все метки, созданные другими пользователями, а также удалять собственные.

Веб-приложение написано с помощью фреймворка Django. Бекэнд машинного обучения - Keras. Обучаемая модель последовательная (sequential). Данные для обучения, валидации и тестирования взяты с сервиса iNaturalist.

В ветке [`main`](https://github.com/sueta1995/CourseWork_TRPS/tree/main) находятся отчет *(название файла)*, техническовое задание *(название файла)*, документация *(название файла)*. В ветке [`machine_learing`](https://github.com/sueta1995/CourseWork_TRPS/tree/machine_learning) располагаются блокноты Jupyter Notebook с функциями для загрузки и подготовки данных для обучения и создания и обучения модели. В ветке [`web`]() находится само веб-приложение.

## Этапы работы

Работа глобально разделена на две части: создание датасета из загруженных данных о стрекозах и создание и обучение модели на основе этих данных, создание веб-приложения.

- [x] **16.06.2023-20.06.2023** - *Загрузка данных с сервиса iNaturalist, анализ данных, создание из загруженных данных датасета.*
- [x] **16.07.2023-22.07.2023** - *Изучение существующих архитектур моделей, изучение сверточных нейронных сетей.*
- [x] **23.07.2023-26.07.2023** - *Написание кода модели, попытка обучения модели и ее тестирование на тестируемых данных, исследование использованных вариантов.*
- [ ] **27.07.2023-02.08.2023** - *Изучение фреймворка Django, повторение паттерна MVC, для создания веб-приложений.*
- [ ] **03.08.2023-06.08.2023** - *Создание главного меню и системы регистрации.*
- [ ] **07.08.2023-17.08.2023** - *Внедрение API карт в веб-приложение, реализация функции создания меток с найденными пользователями стрекозами, внедрение модели для предсказания вида.*
- [ ] **18.08.2023-20.08.2023** - *Тестирование первой версии созданной программы.*

~~А потом со слезами читать книгу про сети и готовиться к Тихомировой.~~

## Загрузка данных и создание датасета

Сначала с помощью сервиса [iNaturalist](https://www.inaturalist.org/observations/export) были загружены csv файлы, в которых содержатся датафреймы с двумя полями `scientific_name` и `image_url` (ссылка на файловый сервер с фотографией стрекозы) и которые соответствуют видам, обитающим в Москве. Данные о видах стрекоз, обитающих в Москве и МО, взяты из [проекта "Стрекозы Московской области/Dragonflies of the Moscow region (Russia)"](https://www.inaturalist.org/projects/strekozy-moskovskoy-oblasti-dragonflies-of-the-moscow-region-russia-815bf252-ee71-4519-840f-1620fd207bc5). Загружены данные для 64 видов.

Сначала был создан скрипт для загрузки фотографий с сервиса iNaturalist для каждого из 64 видов. Ниже представлен код для чтения всех файлов `observation-*.csv` и объединения их в один датафрейм Pandas.

```python
df_columns = ['image_url', 'scientific_name']
df = pd.DataFrame(columns=df_columns)

for filename in glob.glob(f"{path}\\*.csv"):
    temp_df = pd.read_csv(filename)
    df = pd.concat([df, temp_df], axis=0)
    
df.index = range(len(df))
```

После подготовки датафрейма были созданы директории для каждого вида стрекозы и загружены фотографии по полю `image_url` (код ниже, загрузка происходит с 96936-го элемента, так как при загрузке произошла ошибка, и выполнение программы прекратилось).

```python
sub_df = df.iloc[96935:]
sub_df

index = 96936

for raw in sub_df.iloc:
    try:
        filename = f"{index}.jpg"

        urllib.request.urlretrieve(raw['image_url'], filename)
        shutil.move(filename, f"\\mnt\\d\\odonata\\{raw['scientific_name']}")

        index += 1
    except Exception:
        continue
```

Проверка созданных директорий в ходе выполнения программы, код проверки представлен ниже.

```python
DATADIR = "/mnt/d/odonata"
CATEGORIES = os.listdir(DATADIR)

print(f"{len(CATEGORIES)}\n{CATEGORIES}")
```

Вывод:

```
64
['Aeshna affinis', 'Aeshna crenata', 'Aeshna cyanea', 'Aeshna grandis', 'Aeshna isoceles', 'Aeshna juncea', 'Aeshna mixta', 'Aeshna soneharai', 'Aeshna subarctica', 'Aeshna viridis', 'Anax imperator', 'Anax parthenope', 'Brachytron pratense', 'Calopteryx splendens', 'Calopteryx virgo', 'Coenagrion armatum', 'Coenagrion hastulatum', 'Coenagrion johanssoni', 'Coenagrion lunulatum', 'Coenagrion ornatum', 'Coenagrion puella', 'Coenagrion pulchellum', 'Cordulia aenea', 'Enallagma cyathigerum', 'Epitheca bimaculata', 'Erythromma najas', 'Erythromma viridulum', 'Gomphus vulgatissimus', 'Ischnura elegans', 'Ischnura pumilio', 'Lestes barbarus', 'Lestes dryas', 'Lestes sponsa', 'Lestes virens', 'Leucorrhinia albifrons', 'Leucorrhinia caudalis', 'Leucorrhinia dubia', 'Leucorrhinia pectoralis', 'Leucorrhinia rubicunda', 'Libellula depressa', 'Libellula fulva', 'Libellula quadrimaculata', 'Nehalennia speciosa', 'Onychogomphus forcipatus', 'Ophiogomphus cecilia', 'Orthetrum brunneum', 'Orthetrum cancellatum', 'Orthetrum coerulescens', 'Pantala flavescens', 'Platycnemis pennipes', 'Pyrrhosoma nymphula', 'Somatochlora arctica', 'Somatochlora flavomaculata', 'Somatochlora metallica', 'Stylurus flavipes', 'Sympecma fusca', 'Sympecma paedisca', 'Sympetrum danae', 'Sympetrum flaveolum', 'Sympetrum fonscolombii', 'Sympetrum pedemontanum', 'Sympetrum sanguineum', 'Sympetrum striolatum', 'Sympetrum vulgatum']
```

Из вывода понятно, что все директории созданы верно, а их количество соответствует общему количеству видов стрекоз, которые рассматриваются в рамках проекта. Теперь необходимо проверить правильность загрузки фотографий, код этой проверки представлен ниже.

```python
check = []

for category in CATEGORIES:
    check.append((category, len(os.listdir(os.path.join(DATADIR, category)))))

print(check)
```

Вывод проверки:

```
[('Aeshna affinis', 947), ('Aeshna crenata', 276), ('Aeshna cyanea', 4717), ('Aeshna grandis', 2239), ('Aeshna isoceles', 1510), ('Aeshna juncea', 2233), ('Aeshna mixta', 4545), ('Aeshna soneharai', 426), ('Aeshna subarctica', 462), ('Aeshna viridis', 258), ('Anax imperator', 3808), ('Anax parthenope', 2016), ('Brachytron pratense', 1018), ('Calopteryx splendens', 6878), ('Calopteryx virgo', 4116), ('Coenagrion armatum', 317), ('Coenagrion hastulatum', 1687), ('Coenagrion johanssoni', 366), ('Coenagrion lunulatum', 266), ('Coenagrion ornatum', 79), ('Coenagrion puella', 6972), ('Coenagrion pulchellum', 3514), ('Cordulia aenea', 2143), ('Enallagma cyathigerum', 6316), ('Epitheca bimaculata', 667), ('Erythromma najas', 3220), ('Erythromma viridulum', 2127), ('Gomphus vulgatissimus', 2440), ('Ischnura elegans', 5329), ('Ischnura pumilio', 1146), ('Lestes barbarus', 1372), ('Lestes dryas', 3252), ('Lestes sponsa', 2903), ('Lestes virens', 1303), ('Leucorrhinia albifrons', 456), ('Leucorrhinia caudalis', 341), ('Leucorrhinia dubia', 980), ('Leucorrhinia pectoralis', 761), ('Leucorrhinia rubicunda', 1281), ('Libellula depressa', 3896), ('Libellula fulva', 3601), ('Libellula quadrimaculata', 4790), ('Nehalennia speciosa', 190), ('Onychogomphus forcipatus', 3332), ('Ophiogomphus cecilia', 817), ('Orthetrum brunneum', 2961), ('Orthetrum cancellatum', 4121), ('Orthetrum coerulescens', 3634), ('Pantala flavescens', 4015), ('Platycnemis pennipes', 4013), ('Pyrrhosoma nymphula', 4359), ('Somatochlora arctica', 182), ('Somatochlora flavomaculata', 582), ('Somatochlora metallica', 868), ('Stylurus flavipes', 313), ('Sympecma fusca', 3577), ('Sympecma paedisca', 1938), ('Sympetrum danae', 3576), ('Sympetrum flaveolum', 3307), ('Sympetrum fonscolombii', 3987), ('Sympetrum pedemontanum', 1577), ('Sympetrum sanguineum', 6955), ('Sympetrum striolatum', 3880), ('Sympetrum vulgatum', 4333)]
```

Для каждого из видов фотографии загружены верно. Теперь можно переходить к подготовке этих загруженных данных для обучения и созданию модели.

Весь код, отвечающий за загрузку данных и создание датасета, располагается в [`classification/load_photos.ipynb`](https://github.com/sueta1995/CourseWork_TRPS/blob/machine_learning/classification/load_photos.ipynb).

## Подготовка данных для обучения модели

Основная проблема при подготовке данных для обучения - сильная их неравномерность. Например, количество фотографий для вида *Coenagarion lumalatum* (стрелка весенняя) - 79, а для вида *Aeshna cyeanea* (коромысло синее, ~~мой самый любимый вид~~) - 4717. Для наиболее качественного обучение необходимо, чтобы данные были равномерно распределены, то есть для каждого из видов. Поэтому было принято решение не брать в рассмотрение редкие виды стрекоз, для которых количество фотографий меньше 1000. 

Код, который реализует это, представлен ниже.

```python
DATADIR = "/mnt/d/odonata"
CATEGORIES = np.array(os.listdir(DATADIR))

CATEGORIES = np.fromiter(
    (category for category in CATEGORIES if len(os.listdir(os.path.join(DATADIR, category))) > 1000),
    dtype = CATEGORIES.dtype
)

print(len(CATEGORIES))
```

Вывод:

```
44
```

Это означает, что будущая модель будет обучаться на основе 44 классов, в каждом из которых по 1000 данных для обучения и валидации.

Далее необходимо распределить данные на данные для тренировки и данные для валидации, разместить это все в отдельном каталоге. В каждом классе будет 900 фотографий для тренировки и 100 фотографий для валидации.

Код, который реализует разделение данных для тренировки, представлен ниже.

```python
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        dst_path = f"/mnt/d/odonata_data/train_data/{category}"
        
        os.mkdir(dst_path)
        
        for _ in tqdm(range(900)):
            try:
                img_path = os.path.join(path, os.listdir(path)[_])
                
                shutil.copy(img_path, dst_path)
            except Exception as e:
                pass
```

И код для разделения данных для валидации:

```python
def create_validation_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        dst_path = f"/mnt/d/odonata_data/valid_data/{category}"
        
        os.mkdir(dst_path)
        
        for _ in tqdm(range(900, 1000)):
            try:
                img_path = os.path.join(path, os.listdir(path)[_])
                
                shutil.copy(img_path, dst_path)
            except Exception as e:
                pass
```

Библиотека `tqdm` необходима для того, чтобы отслеживать каждую итерацию во вложенном цикле.

После вызова функций `create_training_data` и `create_validation_data` были созданы директории `train_data` и `valid_data` соответственно, и в каждой из них находились по 44 папки, соответствующие всем классам. В каждой директории внутри `train_data` находилось 900 фотографий, внутри `valid_data` - 100.

Весь код, отвечающий за подготовку данных для обучения модели, располагается в [`classification/odonata_data.ipynb`](https://github.com/sueta1995/CourseWork_TRPS/blob/machine_learning/classification/odonata_data.ipynb).

## Создание модели и ее обучение

## Главное меню приложения и система регистрации

## Функция создания меток со стрекозами
