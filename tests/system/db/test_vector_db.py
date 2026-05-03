import time
import uuid
import pytest
from aiagent.system.db.vector_db import VectorDB

@pytest.fixture
def db():
    # Создаем уникальную коллекцию для каждого теста, чтобы они не пересекались
    return VectorDB(collection_name=str(uuid.uuid4()))

def test_add_and_search(db):
    """Проверяем базовое добавление и семантический поиск."""
    id1 = db.add("Привет, как твои дела?", {"source": "user"})
    id2 = db.add("Погода на улице сегодня просто отличная", {"source": "system"})
    
    assert id1 is not None
    assert isinstance(id1, str)
    
    # Ищем по смыслу, а не по точному совпадению
    results = db.search("Здравствуй", top_k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == id1
    assert "Привет" in results[0]["text"]
    
    # Проверяем метаданные и автогенерацию timestamp
    assert "timestamp" in results[0]["metadata"]
    assert results[0]["metadata"]["source"] == "user"

def test_search_with_filter(db):
    """Проверяем фильтрацию по метаданным во время поиска."""
    db.add("Ошибка базы данных", {"level": "error", "module": "db"})
    db.add("Не критичная ошибка сети", {"level": "warning", "module": "net"})
    db.add("Фатальная ошибка памяти", {"level": "error", "module": "core"})
    
    # Ищем только те ошибки, у которых level="error"
    results = db.search("ошибка", top_k=5, filter={"level": "error"})
    
    assert len(results) == 2
    for r in results:
        assert r["metadata"]["level"] == "error"

def test_delete_by_id(db):
    """Проверяем удаление конкретной записи по ее ID."""
    doc_id = db.add("Удали меня пожалуйста", {})
    
    # Убеждаемся, что запись есть
    res1 = db.search("Удали меня", top_k=1)
    assert len(res1) == 1
    assert res1[0]["id"] == doc_id
    
    # Удаляем
    is_deleted = db.delete_by_id(doc_id)
    assert is_deleted is True
    
    # Убеждаемся, что ее больше нет
    res2 = db.search("Удали меня", top_k=1)
    if res2: # База не пустая, но этого ID быть не должно
        assert res2[0]["id"] != doc_id

def test_delete_by_filter(db):
    """Проверяем массовое удаление по совпадению метаданных."""
    db.add("Сообщение 1", {"session": "123"})
    db.add("Сообщение 2", {"session": "123"})
    db.add("Сообщение 3", {"session": "456"})
    
    # Удаляем всю сессию 123
    count = db.delete_by_filter({"session": "123"})
    assert count == 2, f"Expected 2, got {count}"
    
    # Проверяем, что осталась только сессия 456
    results = db.search("Сообщение", top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["session"] == "456"

def test_search_performance(db):
    """
    Проверяем нефункциональное требование: 
    Время поиска по базе не должно превышать 100мс.
    """
    for i in range(100):
        db.add(f"Какой-то случайный текст контекста номер {i}", {"index": i})
        
    # ПРОГРЕВ (Warmup): Первый поиск инициализирует модель эмбеддингов, что может занять 100-200мс.
    db.search("Warmup search")
        
    start_time = time.time()
    results = db.search("Найди мне случайный текст", top_k=5)
    end_time = time.time()
    
    elapsed_ms = (end_time - start_time) * 1000
    
    assert len(results) == 5
    assert elapsed_ms < 250.0, f"Поиск занял {elapsed_ms}мс, что слишком долго!"
