# Assignment 2 — PPO для робо-такси

**Автор:** Карим Хабибрахманов / Karim Khabibrakhmanov  
**Курс:** [F26] Introduction to Adaptive and Efficient Machine Learning, Innopolis

Обучение нейронной политики на JAX в среде Assignment 1. Текущая редакция среды записана в метаданных запусков как `consistent_geometry_v1`.
Успех требует **остановки**: расстояние ≤1, скорость ≤0.001, без столкновения.
Actor получает только 70 чисел наблюдения v2. Полное состояние, экспертный
контроллер и будущая траектория трафика ему не передаются.

Материалы: [отчёт](REPORT.md), [видео](artifacts/demo.mp4),
[текущая перепроверка](artifacts/recheck/summary.json), [исходные эксперименты](artifacts/evaluation/summary.json),
[графики](artifacts/figures/learning.png), [проверки](artifacts/tests.txt).
Точные настройки и обучающие CSV находятся рядом с каждым checkpoint.
`training.csv` содержит обучающую награду с включёнными добавками; таблицы качества используют исходную награду среды. Демонстрации A1 с эвристическим контроллером проверяют саму среду.

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
вместо `python`. Скрипты сами выбирают CPU. [CI исходной версии](https://github.com/Karimkhab/adaptive-and-efficient-ml/actions/runs/41421653274)
прошёл на Linux и Windows. На каждой ОС прошёл 51 тест, одно обновление обучения, 128/128 стартов карты 2, 8/8 граничных случаев и запись видео. Доказательства
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
| 1 | `artifacts/ppo_v2_map1_rechecked/checkpoint.npz` | Перенос с карты 0, затем 150 дополнительных обновлений в исправленной среде, управление 10 Гц |
| 2 | `artifacts/ppo_v2_map2_rechecked/checkpoint.npz` | 350 дополнительных обновлений в исправленной среде, управление 7.5 Гц |
| 3 | `artifacts/ppo_v2_map3_seed0/checkpoint_1000.npz` | PPO с нуля |

Это отдельные политики для четырёх фиксированных карт. Результат одной модели
на всех картах отдельно измеряется как zero-shot transfer.

## Воспроизвести измерения без обучения

```bash
python src/evaluation_suite.py --final-only --summary-only --output artifacts/recheck-reproduced
python src/benchmark.py --repeats 9
python src/make_rl_video.py
python src/report.py
```

Первая команда повторяет исходные диапазоны seed в исправленной среде с актуальными
весами. Для карты 2 это 50000–50511, для остальных карт 20000–20127. Записываются
парковки, столкновения и таймауты для детерминированных и случайных действий,
канонического старта и восьми граничных вариантов. `--heading-noise` задаёт
максимальное отклонение курса в радианах при обычной оценке. Без этого флага
используется ±10°.

`report.py` сверяет таблицы и ссылки отчёта с сохранёнными измерениями.
`benchmark.py` отделяет компиляцию от прогретых запусков и сверяет состояние,
суммы наблюдений и наград и число сигналов завершения при одинаковой работе.
Ролик идёт в 2 раза быстрее симуляции, с явной меткой на экране. Он показывает
фиксированные канонические старты и не заменяет таблицу всех исходов.

## Повтор оценки с физикой float64

`JAX_ENABLE_X64=1` включает 64-битные вычисления симулятора. Готовая модель
по-прежнему получает наблюдения `float32` и использует сохранённые веса.
Для карты 2 на тех же 512 стартах:

```bash
JAX_ENABLE_X64=1 python src/evaluate_rl.py --checkpoint artifacts/ppo_v2_map2_rechecked/checkpoint.npz --maps 2 --episodes 512 --seed 50000 --output artifacts/evaluation/map2-float64.json
```

В PowerShell сначала задайте `$env:JAX_ENABLE_X64 = "1"`, затем выполните
команду Python без префикса. Для исходного режима установите значение `0`.
[Сравнение на всех картах](artifacts/precision_comparison.json).

## Обучить с нуля

Минимальный самостоятельный эксперимент:

```bash
python src/train.py --map 0 --randomize --route-reward --parking-reward --updates 500 --output artifacts/my_run
python src/evaluate_rl.py --checkpoint artifacts/my_run/checkpoint_0500.npz --maps 0 --episodes 128 --seed 20000 --output artifacts/my_run/evaluation.json
```

Повтор дообучения после исправления среды:

```bash
python src/train.py --map 1 --seed 21 --updates 225 --randomize --route-reward --parking-reward --clearance-speed --clearance-cost 4 --clearance-margin 1 --cruise-speed 3.5 --speed-cost .4 --learning-rate .00007 --entropy 0 --init-checkpoint artifacts/ppo_v2_map1_transfer_seed0/checkpoint_1000.npz --output artifacts/reproduced-map1
python src/train.py --map 2 --seed 12 --updates 400 --randomize --route-reward --parking-reward --clearance-speed --clearance-cost 5 --clearance-margin 1.1 --cruise-speed 2.8 --speed-cost .75 --learning-rate .00015 --entropy .001 --init-checkpoint artifacts/ppo_v2_map2_seed0/checkpoint_1600.npz --output artifacts/reproduced-map2
```

В итоговых метаданных указано `inference_repeat=3` для checkpoint 150 карты 1 и `inference_repeat=4` для checkpoint 350 карты 2. Если оцениваете собственные веса, задайте это поле в JSON рядом с соответствующим NPZ. Готовые файлы выше уже содержат эти значения.

Параметры исходной матрицы можно запустить в исправленной среде, задав **новый** каталог:

```bash
python src/train_all.py --root artifacts/reproduced
python src/evaluation_suite.py --root artifacts/reproduced
```

Для точного контекста исходных таблиц используйте [версию от 2 ноября](https://github.com/Karimkhab/adaptive-and-efficient-ml/tree/5f0ace8/assignment-2). Для сравнения с исходной наградой уберите `--route-reward --parking-reward`.
Существующий обучающий каталог защищён от случайной перезаписи. Веса позволяют
полностью повторить inference; checkpoints не содержат состояния оптимизатора
и не являются точным снимком для побитового продолжения прерванного обучения.
Полная матрица требует существенно больше времени, чем оценка готовых моделей;
измеренное время циклов обучения приведено в отчёте. Межплатформенная побитовая
идентичность обучения не предполагается.

Источник среды и рендерера — Assignment 1 Карима и исходная заготовка
[InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment).
Ни публикация, ни сообщение преподавателю этим комплектом не выполняются.
