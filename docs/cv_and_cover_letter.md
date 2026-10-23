# bala-qlora — формулировки для CV и письма

Автор проекта: **Арыстан Қабдеш (Arystan Kabdesh)**.
Ниже — готовые формулировки от первого лица об опыте разработки проекта.

## CV — английская версия

**bala-qlora — Bilingual QLoRA Fine-Tuning Experiment**

Python · PyTorch · Transformers · PEFT · QLoRA · pytest · Git · Google Colab

- Built an AI-assisted QLoRA pipeline for Qwen2.5-1.5B-Instruct using 310 Kazakh/Russian examples, including 300 synthetic drafts; trained an 18.5M-parameter adapter on a free Colab T4.
- Implemented topic-disjoint train/validation/test splits, answer-only loss masking, SHA-256 data checks and 43 automated tests; evaluated the base and adapted models on 32 held-out examples and packaged the adapter, inference code and experiment evidence.

Короткая версия, если в CV мало места:

> Developed and tested a bilingual QLoRA pipeline with 310 examples, 43 automated tests, real T4 training and a controlled 32-example base-versus-adapter evaluation.

## Сопроводительное письмо — английская версия

> In my bala-qlora project, I used Claude Code and Codex to build and evaluate a Kazakh/Russian QLoRA pipeline. I took the experiment from synthetic data preparation through 43 automated tests, training on a free Colab T4, and a controlled comparison with the base model. I also resolved a Git line-ending issue that broke dataset checksums across Windows and Linux. Although text-similarity and formatting metrics improved, the content review exposed factual errors and weak Kazakh generation. I documented those limits and packaged the adapter, source snapshots and results so the work can be inspected and repeated.

## CV — русская версия

- Разработал QLoRA-пайплайн для Qwen2.5-1.5B-Instruct на 310 казахских и русских примерах, включая 300 синтетических черновиков, с использованием ИИ-инструментов; обучил адаптер с 18,5 млн параметров на бесплатном Colab T4.
- Реализовал разбиение по темам без пересечения переводов, loss только по ответам, контроль SHA-256 и 43 автотеста. Сравнил базовую модель и адаптер на 32 отложенных примерах, сохранил веса, код инференса и доказательства запуска.

## Краткий фидбек

Проект подтверждает практический опыт построения и проверки небольшого
ML-эксперимента: работа с данными, PyTorch/Transformers/PEFT, QLoRA,
тестирование, Git, отладка и анализ результатов. Самый сильный акцент для
Junior Agentic Developer — проверка результата работы ИИ-инструментов
и доведение эксперимента до сохранённых, проверяемых артефактов.

Рост chrF **18,88 → 25,09** можно обсуждать на интервью только как рост
сходства с эталоном. Слепая ИИ-проверка содержания дала QLoRA 5 предпочтений,
базовой модели 6 и 21 ничью; все 16 казахских пар признаны проблемными.
Доказанного общего улучшения качества и готового детского продукта нет.

## Что объяснить на интервью

1. Зачем темы и их переводы держать в одной выборке: иначе близкие ответы из train могут завысить тестовую оценку.
2. Почему маскируется промпт: обучающий loss должен оценивать генерацию ответа.
3. Что делает QLoRA: базовая модель загружается в 4 битах, обучаются небольшие адаптерные матрицы.
4. Почему нужен baseline с одинаковой генерацией: различия настроек не должны подменять эффект дообучения.
5. Почему рост chrF не равен правильности: бессвязный ответ тоже может совпадать с эталоном по символам.
6. Как обнаружена проблема CRLF/LF: SHA-256 менялся после Git-экспорта; точные байты защищены правилами `.gitattributes` и проверены в свежем архиве.

В документации сохранено происхождение синтетических данных и ИИ-проверки.
Это позволяет говорить о проекте как о своём, не приписывая себе
независимую ручную проверку или экспертное подтверждение качества.

Репозиторий пока локальный, без GitHub push. Для демонстрации используйте
выполненный ноутбук и архив. Доступ к Colab для работодателя настраивает
владелец; наличие ссылки само по себе не означает публичный доступ.
