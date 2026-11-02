# Assignment 2 — PPO для робо-такси

**Автор:** Карим Хабибрахманов / Karim Khabibrakhmanov  
**Курс:** [F26] Introduction to Adaptive and Efficient Machine Learning, Innopolis

Обучение нейронной политики на JAX в актуальной среде Assignment 1
([`9f015e2`](https://github.com/Karimkhab/adaptive-and-efficient-ml/commit/9f015e2cbadbfb24ea54a894d8db3e4b552c0554)).
Успех требует **остановки**: расстояние ≤1, скорость ≤0.001, без столкновения.
Actor получает только 70 чисел наблюдения v2. Полное состояние, экспертный
контроллер и будущая траектория трафика ему не передаются.

Материалы: [отчёт](REPORT.md), [видео](artifacts/demo.mp4),
[результаты по каждому эпизоду](artifacts/evaluation/summary.json),
[графики](artifacts/figures/learning.png), [проверки](artifacts/tests.txt).
Точные настройки и обучающие CSV находятся рядом с каждым checkpoint.
`training.csv` содержит обучающую награду с включёнными добавками; итоговые
таблицы качества используют исходную награду из `evaluation/`.
Демонстрации A1 с эвристическим контроллером используются только для аудита среды;
числа RL относятся к `evaluation/`.

## Установка

Python **3.12**, CPU, без Docker/CUDA и внешних сервисов.
Команды выполняются из этого каталога (`assignment2` или `assignment-2` после копирования).

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip check
```

При запрете активации PowerShell можно вызывать `.\.venv\Scripts\python.exe`
вместо `python`. Скрипты сами выбирают CPU. [Настоящий CI A2](https://github.com/Karimkhab/adaptive-and-efficient-ml/actions/runs/41421653274)
прошёл Linux и Windows: на каждой ОС51 тест, update обучения,128/128 новых
стартов map2,8/8 endpoint cases и запись/декодирование видео. Доказательства
и точные версии находятся в отчёте и `artifacts/ci_native/`.

## Проверить и запустить готовую модель

```bash
python -m pytest -q
python src/play.py --checkpoint artifacts/ppo_v2_map3_seed0/checkpoint_1000.npz
python src/play.py --checkpoint artifacts/ppo_v2_map0_seed0/checkpoint_1500.npz --headless --randomize
```

В окне: Z — масштаб, L — лучи, C — коллайдеры, Esc — выход.
После завершения эпизода скрипт печатает исход, расстояние, скорость и return.
Для любой модели обязательны оба одноимённых файла: `.npz` с весами и `.json`
с архитектурой/контрактом и настройками.

| Карта | Checkpoint | Подготовка |
| --- | --- | --- |
| 0 | `artifacts/ppo_v2_map0_seed0/checkpoint_1500.npz` | PPO с нуля |
| 1 | `artifacts/ppo_v2_map1_transfer_seed0/checkpoint_1000.npz` | PPO из весов map0 после 500 updates, затем 1000 updates на map1 |
| 2 | `artifacts/ppo_v2_map2_final/checkpoint.npz` | PPO1600 +600 safety-updates; inference30 Гц, обучение5 Гц |
| 3 | `artifacts/ppo_v2_map3_seed0/checkpoint_1000.npz` | PPO с нуля |

Это отдельные политики для четырёх фиксированных карт. Результат одной модели
на всех картах отдельно измеряется как zero-shot transfer.

## Воспроизвести измерения без обучения

```bash
python src/evaluation_suite.py
python src/benchmark.py
python src/charts.py
python src/make_rl_video.py
python src/report.py
```

`report.py` сверяет ключевые таблицы и ссылки с сохранёнными измерениями.
Объясняющий текст отчёта редактируется вручную; скрипт его не перезаписывает.

Оценка использует замороженные checkpoints: по128 reset seeds20000–20127 для карт0/1/3,
а для доработанной карты2 —512 новых seeds50000–50511. Старые128 стартов map2
теперь регрессионные. Проверяются
канонические старты, восемь граничных вариантов на карту, стохастические действия,
baseline и проверки переноса. Для проверки разработки использовались другие
seeds (для map2 —10000–10511). Карты известны при обучении; это проверка новых условий
старта/трафика, а не новых карт. Метрика — парковка, отдельно collision и timeout;
return всегда исходный, без обучающих добавок к награде.

`benchmark.py` отделяет первый вызов с компиляцией от пяти прогретых запусков,
синхронизирует JAX и сверяет конечные состояния при одинаковом числе переходов.
Ролик идёт в 2 раза быстрее симуляции, с явной меткой на экране. Он показывает
фиксированные канонические старты и не заменяет таблицу всех исходов.

## Обучить с нуля

Минимальный самостоятельный эксперимент:

```bash
python src/train.py --map 0 --randomize --route-reward --parking-reward --updates 500 --output artifacts/my_run
python src/evaluate_rl.py --checkpoint artifacts/my_run/checkpoint_0500.npz --maps 0 --episodes 128 --seed 20000 --output artifacts/my_run/evaluation.json
```

Полная матрица из отчёта в **новый** каталог:

```bash
python src/train_all.py --root artifacts/reproduced
python src/evaluation_suite.py --root artifacts/reproduced
```

Для сравнения с исходной наградой уберите `--route-reward --parking-reward`.
Существующий обучающий каталог защищён от случайной перезаписи. Веса позволяют
полностью повторить inference; checkpoints не содержат состояния оптимизатора
и не являются точным снимком для побитового продолжения прерванного обучения.
Полная матрица требует существенно больше времени, чем оценка готовых моделей;
измеренное время циклов обучения приведено в отчёте. Межплатформенная побитовая
идентичность обучения не предполагается.

Источник среды и рендерера — Assignment 1 Карима и исходная заготовка
[InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment).
Ни публикация, ни сообщение преподавателю этим комплектом не выполняются.
