# Assignment 1 — среда автомобиля на JAX

**Автор:** Карим Хабибрахманов / Karim Khabibrakhmanov  
**Курс:** [F26] Introduction to Adaptive and Efficient Machine Learning, Innopolis

Автомобиль разгоняется, тормозит и поворачивает, едет к цели среди статических и
движущихся препятствий. Среда считает наблюдения, награду и столкновения.
Автоматическая демонстрация использует обычный контроллер; RL-обучение — во второй ассиге.

[Отчёт](REPORT.md) · [Видео](artifacts/demo.mp4) · [Результаты](artifacts/evaluation.json)

## Установка

Проверено на Python 3.12.14, macOS arm64, JAX CPU. В терминале:

```bash
git clone --branch Submission https://github.com/Karimkhab/adaptive-and-efficient-ml.git
cd adaptive-and-efficient-ml/assignment-1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

## Запуск

```bash
python src/main.py --map 0          # управление с клавиатуры
python src/main.py --map 2 --demo   # автоматическая поездка
```

W/↑ — газ, S/↓ — тормоз, A/← и D/→ — руль, R — новый эпизод, Esc — выход.
Карты 0–2 взяты из заготовки; карта 3 проверяет пересечение с движущимся препятствием.
В окне синий круг — машина, зелёный — цель, серые — статические препятствия,
красные — движущиеся. После столкновения, достижения цели или таймаута эпизод завершается.

## Проверки и запись видео

```bash
JAX_PLATFORM_NAME=cpu python -m pytest -q
JAX_PLATFORM_NAME=cpu python src/evaluate.py
python src/make_video.py
```

Результаты сохраняются в `artifacts/`: JSON, CSV траекторий, кадры и `demo.mp4`.
Зафиксированы 24 пройденных теста и четыре поездки без столкновений. Сценарии
детерминированы; это проверка среды, а не оценка обученного агента на новых картах.

## Где код

- `src/env/robotaxi.py` — динамика, состояния, действия, наблюдения, награда и завершение.
- `src/controller.py` — эвристический водитель для демонстрации, использующий полное состояние.
- `src/main.py`, `src/utils/taxi_renderer.py` — запуск и Pygame-рендерер.
- `tests/test_robotaxi.py` — проверки физики, интерфейса и полных поездок.

## Источники

Заготовка: [InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment/tree/1056d711e166747cc485725e31f2535967b030dd),
коммит `1056d711e166747cc485725e31f2535967b030dd`. Сохранены исходные `base.py`,
`tasks.py`, `utils/renderer.py` и `utils/utils.py`; `RobotaxiEnv` наследует `BaseEnv`.
Добавлены модель машины, контроллер, адаптированный рендерер и проверки; заменён
`main.py`, а в `autoreset.py` исправлено хранение начального состояния.

Модель: [Robotics Toolbox — Bicycle](https://petercorke.github.io/robotics-toolbox-python/mobile-vehicle-bicycle.html).
Требования: Assignments.pdf от 30 сентября 2026 и [Stage 1](https://github.com/InnoRL/IAEML-assignment/blob/1056d711e166747cc485725e31f2535967b030dd/instructions/stage_1.md).
