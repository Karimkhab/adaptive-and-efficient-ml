# Материалы Assignment 2

| Что хочется посмотреть | С чего начать |
| --- | --- |
| Как едут готовые модели | [Видео четырёх парковок](demo.mp4) · [Исходы показанных поездок](demo.json) |
| Сколько поездок успешны | [График текущих результатов](figures/recheck.png) · [Данные по каждому старту](recheck/summary.json) |
| Как PPO ведёт себя на незнакомой карте | [Схема карты 4, результаты и исходы по seed](map4_validation/README.md) |
| Как шло дообучение | [Кривые для карт 1 и 2](figures/finetuning.png) · [Сравнение кандидатов](recheck/selection.json) |
| Что показали исходные эксперименты | [Кривые обучения](figures/learning.png) · [Сравнение наград](figures/evaluation.png) · [Траектории](figures/trajectories.png) · [Исходы](evaluation/summary.json) |
| Как измерены точность и скорость | [float32 и float64](precision_comparison.json) · [Бенчмарк шага среды](benchmark.json) · [Время сбора опыта](recheck/reset_performance.json) |
| Какие проверки прошли | [Локальные тесты](tests.txt) · [Проверка парковки](recheck/ci-local.json) · [CI на Ubuntu и Windows](https://github.com/Karimkhab/adaptive-and-efficient-ml/actions/runs/42496985643) |

<a name="weights"></a>

## Готовые веса по картам

Для запуска нужны оба файла из строки таблицы. В `.npz` лежат параметры сетей, в `.json` — настройки модели и частота управления. Пары уже лежат рядом в репозитории.

| Карта | Веса | Метаданные | Журнал обучения |
| --- | --- | --- | --- |
| 0 | [NPZ](ppo_v2_map0_seed0/checkpoint_1500.npz) | [JSON](ppo_v2_map0_seed0/checkpoint_1500.json) | [CSV](ppo_v2_map0_seed0/training.csv) |
| 1 | [NPZ](ppo_v2_map1_rechecked/checkpoint.npz) | [JSON](ppo_v2_map1_rechecked/checkpoint.json) | [CSV дообучения](ppo_v2_map1_rechecked/training.csv) |
| 2 | [NPZ](ppo_v2_map2_rechecked/checkpoint.npz) | [JSON](ppo_v2_map2_rechecked/checkpoint.json) | [CSV дообучения](ppo_v2_map2_rechecked/training.csv) |
| 3 | [NPZ](ppo_v2_map3_seed0/checkpoint_1000.npz) | [JSON](ppo_v2_map3_seed0/checkpoint_1000.json) | [CSV](ppo_v2_map3_seed0/training.csv) |

[Машиночитаемый список моделей](final_models.json) · [Команды запуска](../REPORT.md#reproduce)

В `recheck/` лежит повторная проверка текущих моделей, а в `evaluation/` — исходные эксперименты. Файлы `demo_map*.csv` содержат покадровые данные поездок из видео. В остальных папках `ppo_v2_*` сохранены промежуточные веса и варианты обучения для сравнения. [Описание экспериментов](../REPORT.md#experiments)
