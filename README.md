# Структура
```
AIAgent/
│
├── config/                             # Статическая конфигурация
│   ├── settings.yaml                   # Общие настройки
│   └── models.yaml                     # Карта путей к весам и параметры моделей
│
├── data/                               # Runtime данные (не в git)
│   ├── agent.db                        # SQL база
│ # └── recordings/                     # Временные wav
│
├── models/                             # ВЕСА НЕЙРОСЕТЕЙ (скачанные файлы)
│ # ├── whisper/
│ # │   └── tiny.pt
│ # ├── translation/
│ # │   └── opus-mt-en-ru/
│ # ├── ...
│   └── README.md                       # Какие версии и где взять
│
├── scripts/                            # Одноразовые скрипты обслуживания
│   ├── download_models.py              # Автоматическая загрузка весов
│ # ├── init_db.py                      # Создание таблиц
│ # └── benchmark_inference.py
│
├── src/
│   └── aiagent/                        # Основной пакет
│       ├── core/                       # Фундамент, конфигурация, события
│       │   ├── __init__.py
│       │   ├── config.py               # Параметры из .env / YAML
│       │   ├── exceptions.py           # Общие ошибки агента
│       │   ├── events.py               # Шина событий (AudioCaptured, TranscriptReady, TextGenerating...)
│       │   └── logging.py
│       │
│       ├── domain/                     # Доменные модели (НЕ ML!)
│       │   ├── __init__.py
│       │   ├── message.py              # Класс Message
│       │   ├── dialogue_state.py       # Класс DialogContext
│       │   └── actions.py              # Команды, которые может выполнить агент
│       │
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── basemodel.py            # Абстрактный класс NeuralModel
│       │   ├── registry.py             # Реестр моделей (синглтон)
│       │   ├── whisper_model.py        # Whisper для транскрипции
│       │   ├── ipa_translate_model.py  # Модель IPA-транскрипции
│       │   └── tts_model.py            # Text-to-Speech (будущий)
│       │
│       ├── skills/                     # Бизнес-навыки агента (бывший services)
│       │   ├── __init__.py
│       │   ├── factory.py         # Главный цикл: слушаю → распознаю → анализирую → отвечаю
│       │   ├── transcription.py        # Логика превращения аудио в текст
│       │   ├── ipa_translation.py      # Перевод текста　в IPA
│       │ # ├── command_handler.py      # Выполняет команды пользователя
│       │   └── speech.py              # Озвучка ответа
│       │
│       ├── system/             # БД, файлы, внешние интерфейсы
│       │   ├── db/
│       │   │   ├── __init__.py
│       │   │   ├── engine.py           # Подключение SQL
│       │   │   └── models.py           # ORM-таблицы (MessageRecord, AgentMemory и т.п.)
│       │   └── micro/
│       │       ├── __init__.py         # Обёртка pyaudio/sounddevice
│       │       ├── capture.py
│       │       └── player.py
│       │
│       ├── utils/                      # Чистые вспомогательные функции
│       │   ├── audio_utils.py          # Ресемплинг, chunking
│       │   └── ...
│       │
│       ├── __init__.py
│       └── main.py                     # Точка входа CLI
│
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md 
```

Я не умею составлять такие файлы (Инфо, ТЗ и т.д.), поэтому получилось так, как получилось.

# Инфо

## О чём проект?
Мы создаём голосового помощника на основе AI, который будет выполнять следующие задачи:
1. Обработка аудио
    - Получение голосового запроса
    - Преобразование STT (Speech To Text)
2. Поиск полученных данных в памяти (база данных)
3. Отправка промпта в LLM -> Получение ответа
4. Обработка аудио
    - Получение текста
    - Преобразование в IPA-транскрипцию
    - Преобразование TTS (Text To Speech)
