# Assignment 1 — Campus Taxi

**Автор:** Карим Хабибрахманов / Karim Khabibrakhmanov  
**Курс:** [F26] Introduction to Adaptive and Efficient Machine Learning, Innopolis

Простая 2D-среда автомобиля на JAX: газ, тормоз, руль, препятствия, цель,
наблюдения и награда. Пиксельный рендерер показывает движение и физические
коллизии; автоматический водитель демонстрационный, без RL-обучения.

[Отчёт](REPORT.md) · [Видео](artifacts/demo.mp4) · [Результаты](artifacts/evaluation.json)

![Демонстрация среды](artifacts/preview_map2.png)

## Установка

Проверено на Python 3.12.14, macOS arm64, JAX CPU. В терминале:

```bash
git clone --branch Submission https://github.com/Karimkhab/adaptive-and-efficient-ml.git
cd adaptive-and-efficient-ml/assignment-1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

## Запуск и управление

```bash
python src/main.py --map 2 --demo   # автоматическая поездка
python src/main.py --map 0          # управление с клавиатуры
```

- **W/↑, S/↓** — газ и тормоз; **A/←, D/→** — руль.
- **Tab** — демо / ручное управление; **Space** — пауза; **R** — новый эпизод.
- **1–4** — выбор карты; **Z** — общий вид / камера машины.
- **L** — лучи датчиков; **C** — круги коллизий; **Esc** — выход.

Карты CLI 0–2 соответствуют раскладкам преподавателя, карта 3 — дополнительное
пересечение. Жёлтая машина — агент, коралловые — движущиеся препятствия,
серо-зелёные — статические; бирюзовые линия и метка показывают путь и цель.
Корпуса нарисованы внутри круглых коллайдеров радиуса 1. Размер клетки — 5.
Камера и пиксельная сетка влияют только на рисунок, а не на состояние среды.

## Проверки и видео

```bash
JAX_PLATFORM_NAME=cpu python -m pytest -q
JAX_PLATFORM_NAME=cpu python src/evaluate.py
python src/make_video.py           # MP4, 1280×800, 30 FPS, воспроизведение 2x
python src/make_video.py --speed 1 # запись в реальном времени
```

В `artifacts/` сохранены 33 пройденных теста, четыре успешные поездки, CSV и видео.
Симуляция всегда работает с шагом 1/30 с; при ускорении видео есть метка 2x,
а таймер показывает время симуляции. Сценарии детерминированы; это проверка среды,
а не оценка обученного агента. Подробности и ограничения — в отчёте.

## Где код и источники

- `src/env/robotaxi.py` — динамика, состояния, действия, наблюдения, награда, коллизии.
- `src/controller.py` — демонстрационный водитель с доступом к полному состоянию.
- `src/main.py`, `src/utils/taxi_renderer.py` — запуск, ввод и рендерер.
- `tests/` — физика, движения препятствий, ручное управление и полные поездки.

Заготовка: [InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment/tree/1056d711e166747cc485725e31f2535967b030dd),
коммит `1056d711e166747cc485725e31f2535967b030dd`. Сохранены исходные `base.py`,
`tasks.py`, `utils/renderer.py` и `utils/utils.py`; новая среда наследует `BaseEnv`.
Модель: [Robotics Toolbox — Bicycle](https://petercorke.github.io/robotics-toolbox-python/mobile-vehicle-bicycle.html).
Для визуального оформления использованы рекомендации [game-graphics](https://github.com/fenre/skills/blob/main/game-graphics/SKILL.md):
маленькие спрайты, ограниченная палитра, увеличение без сглаживания. Графика
нарисована кодом Pygame; внешние изображения и новые графические библиотеки не нужны.
Требования: Assignments.pdf от 30 сентября 2026 и [Stage 1](https://github.com/InnoRL/IAEML-assignment/blob/1056d711e166747cc485725e31f2535967b030dd/instructions/stage_1.md).
