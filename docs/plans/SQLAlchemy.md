# Prompt: Refaktoryzacja systemu zarządzania danymi z obsługą wielu silników baz danych używając SQLAlchemy

## Kontekst
Pracujesz nad refaktoryzacją monolitycznego pliku `duckdb_data_manager.py`, który obecnie jest ściśle powiązany z DuckDB. Celem jest stworzenie architektury używającej **SQLAlchemy Core** jako warstwy abstrakcji, pozwalającej na łatwą wymianę silnika bazy danych między DuckDB, SQLite i PostgreSQL.

## Zadanie główne
Dokonaj refaktoryzacji pliku `duckdb_data_manager.py` używając SQLAlchemy Core zgodnie z następującymi wytycznymi:

### 1. Instalacja i konfiguracja
Najpierw zainstaluj wymagane pakiety:
```bash
pip install sqlalchemy duckdb-engine duckdb psycopg2-binary
```

### 2. Analiza wstępna
- Przeanalizuj obecną strukturę pliku `duckdb_data_manager.py`
- Zidentyfikuj wszystkie miejsca specyficzne dla DuckDB
- Sprawdź plik `@TODO.md` w poszukiwaniu dodatkowych wskazówek dotyczących strategii bazodanowej
- Zapoznaj się z dokumentacją SQLAlchemy Core i duckdb-engine

### 3. Architektura docelowa z SQLAlchemy Core
Zaprojektuj architekturę opartą na następujących zasadach:

**a) Struktura katalogów:**
```
├── database/
│   ├── __init__.py
│   ├── config.py              # Konfiguracja połączeń
│   ├── engine_factory.py      # Factory dla silników SQLAlchemy
│   ├── base_manager.py        # Bazowa klasa używająca SQLAlchemy Core
│   ├── migrations/
│   │   └── migration_tool.py  # Narzędzie do migracji między bazami
│   ├── queries/
│   │   ├── __init__.py
│   │   ├── base_queries.py    # Wspólne zapytania
│   │   └── specific/          # Zapytania specyficzne dla baz
│   │       ├── duckdb_queries.py
│   │       ├── postgres_queries.py
│   │       └── sqlite_queries.py
│   └── utils/
│       ├── type_mapping.py    # Mapowanie typów między bazami
│       └── connection_utils.py # Utilities dla połączeń
```

**b) Przykładowa implementacja z SQLAlchemy Core:**

### 4. Kluczowe wymagania implementacyjne

**a) Konfiguracja silników baz danych:**
```python
# config.py
from sqlalchemy import create_engine
from typing import Dict, Any

class DatabaseConfig:
    @staticmethod
    def get_engine(db_type: str, connection_params: Dict[str, Any]):
        if db_type == 'duckdb':
            # Wymaga duckdb-engine
            conn_string = f"duckdb:///{connection_params.get('database', ':memory:')}"
        elif db_type == 'sqlite':
            conn_string = f"sqlite:///{connection_params.get('database', ':memory:')}"
        elif db_type == 'postgresql':
            user = connection_params.get('user', 'postgres')
            password = connection_params.get('password', '')
            host = connection_params.get('host', 'localhost')
            port = connection_params.get('port', 5432)
            database = connection_params.get('database', 'postgres')
            conn_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        return create_engine(conn_string, **connection_params.get('engine_args', {}))
```

**b) Bazowy manager używający SQLAlchemy Core:**
```python
# base_manager.py
from sqlalchemy import MetaData, Table, select, insert, update, delete, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.metadata = MetaData()
        self._tables = {}
    
    def reflect_table(self, table_name: str) -> Table:
        """Ładuje metadane istniejącej tabeli"""
        if table_name not in self._tables:
            self._tables[table_name] = Table(
                table_name, 
                self.metadata, 
                autoload_with=self.engine
            )
        return self._tables[table_name]
    
    def execute_query(self, query, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Wykonuje zapytanie SELECT używając SQLAlchemy Core"""
        with self.engine.connect() as conn:
            if isinstance(query, str):
                result = conn.execute(text(query), params or {})
            else:
                result = conn.execute(query)
            return [dict(row._mapping) for row in result]
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """Wstawia dane do tabeli"""
        table = self.reflect_table(table_name)
        with self.engine.begin() as conn:
            result = conn.execute(insert(table), data)
            return result.rowcount
    
    def update_data(self, table_name: str, values: Dict[str, Any], 
                   conditions: Dict[str, Any]) -> int:
        """Aktualizuje dane w tabeli"""
        table = self.reflect_table(table_name)
        stmt = update(table).values(**values)
        
        # Dodaj warunki WHERE
        for col, val in conditions.items():
            stmt = stmt.where(getattr(table.c, col) == val)
        
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount
    
    def bulk_insert_from_pandas(self, df, table_name: str, if_exists: str = 'append'):
        """Wykorzystuje metodę to_sql pandas z SQLAlchemy"""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
```

**c) Obsługa specyficznych funkcji baz danych:**
```python
# queries/specific/duckdb_queries.py
from sqlalchemy import text

class DuckDBSpecificQueries:
    @staticmethod
    def read_parquet(engine, file_path: str, table_name: str):
        """DuckDB-specific: czyta plik Parquet"""
        query = text(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{file_path}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
            conn.commit()
    
    @staticmethod
    def export_to_parquet(engine, table_name: str, file_path: str):
        """DuckDB-specific: eksportuje do Parquet"""
        query = text(f"COPY {table_name} TO '{file_path}' (FORMAT PARQUET)")
        with engine.connect() as conn:
            conn.execute(query)
```

**d) Factory pattern dla różnych silników:**
```python
# engine_factory.py
from .config import DatabaseConfig
from .base_manager import DatabaseManager
from .queries.specific import duckdb_queries, postgres_queries, sqlite_queries

class DatabaseFactory:
    @staticmethod
    def create_manager(db_type: str, connection_params: Dict[str, Any]) -> DatabaseManager:
        engine = DatabaseConfig.get_engine(db_type, connection_params)
        manager = DatabaseManager(engine)
        
        # Dodaj metody specyficzne dla danego silnika
        if db_type == 'duckdb':
            manager.read_parquet = lambda fp, tn: duckdb_queries.DuckDBSpecificQueries.read_parquet(engine, fp, tn)
            manager.export_to_parquet = lambda tn, fp: duckdb_queries.DuckDBSpecificQueries.export_to_parquet(engine, tn, fp)
        
        return manager
```

### 5. Przykład użycia po refaktoryzacji

```python
# main.py
from database.engine_factory import DatabaseFactory

# Konfiguracja - łatwa zmiana między bazami
config = {
    'db_type': 'duckdb',  # zmień na 'sqlite' lub 'postgresql'
    'connection_params': {
        'database': './my_database.duckdb'
    }
}

# Utworzenie managera
db = DatabaseFactory.create_manager(**config)

# Użycie - identyczne dla wszystkich baz
results = db.execute_query("SELECT * FROM users WHERE age > :age", {'age': 25})

# Używanie SQLAlchemy Core query builder
from sqlalchemy import select
users_table = db.reflect_table('users')
query = select(users_table).where(users_table.c.age > 25)
results = db.execute_query(query)

# Funkcje specyficzne dla DuckDB (dostępne tylko gdy db_type='duckdb')
if hasattr(db, 'read_parquet'):
    db.read_parquet('data.parquet', 'imported_data')
```

### 6. Strategia migracji

**a) Narzędzie do migracji danych między bazami:**
```python
# migrations/migration_tool.py
from sqlalchemy import inspect, MetaData, Table
import pandas as pd

class DatabaseMigrator:
    def __init__(self, source_engine, target_engine):
        self.source_engine = source_engine
        self.target_engine = target_engine
        self.inspector = inspect(source_engine)
    
    def migrate_table(self, table_name: str, batch_size: int = 10000):
        """Migruje pojedynczą tabelę"""
        # Odczytaj strukturę tabeli
        source_meta = MetaData()
        source_table = Table(table_name, source_meta, autoload_with=self.source_engine)
        
        # Utwórz tabelę w docelowej bazie
        source_table.create(self.target_engine, checkfirst=True)
        
        # Migruj dane partiami
        with self.source_engine.connect() as source_conn:
            with self.target_engine.connect() as target_conn:
                offset = 0
                while True:
                    query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
                    df = pd.read_sql(query, source_conn)
                    
                    if df.empty:
                        break
                    
                    df.to_sql(table_name, target_conn, if_exists='append', index=False)
                    offset += batch_size
    
    def migrate_all_tables(self):
        """Migruje wszystkie tabele"""
        tables = self.inspector.get_table_names()
        for table in tables:
            print(f"Migrating table: {table}")
            self.migrate_table(table)
```

### 7. Najlepsze praktyki zastosowane

1. **SQLAlchemy Core zamiast ORM:**
   - Lepsza wydajność dla operacji analitycznych
   - Większa kontrola nad SQL
   - Łatwiejsza integracja z istniejącym kodem SQL

2. **Wzorce projektowe:**
   - Factory Pattern dla tworzenia odpowiednich managerów
   - Composition over Inheritance - dodawanie funkcji specyficznych dynamicznie
   - Connection pooling automatycznie obsługiwany przez SQLAlchemy

3. **Obsługa różnic między bazami:**
   - Podstawowe operacje używają SQLAlchemy Core (uniwersalne)
   - Funkcje specyficzne są opcjonalne i dodawane dynamicznie
   - Mapowanie typów obsługiwane automatycznie przez SQLAlchemy

4. **Testowanie:**
   ```python
   # tests/test_database_manager.py
   import pytest
   from database.engine_factory import DatabaseFactory
   
   @pytest.mark.parametrize("db_type", ["sqlite", "duckdb", "postgresql"])
   def test_basic_operations(db_type, test_connection_params):
       db = DatabaseFactory.create_manager(db_type, test_connection_params[db_type])
       
       # Test insert
       data = [{'name': 'Test', 'age': 30}]
       rows = db.insert_data('users', data)
       assert rows == 1
       
       # Test query
       results = db.execute_query("SELECT * FROM users WHERE name = :name", {'name': 'Test'})
       assert len(results) == 1
   ```

### 8. Checklist implementacji

- [ ] Zainstaluj wymagane pakiety (SQLAlchemy, duckdb-engine, psycopg2-binary)
- [ ] Analiza obecnego kodu `duckdb_data_manager.py`
- [ ] Utworzenie struktury katalogów
- [ ] Implementacja DatabaseConfig dla konfiguracji połączeń
- [ ] Implementacja DatabaseManager używającego SQLAlchemy Core
- [ ] Implementacja DatabaseFactory
- [ ] Migracja istniejących zapytań do SQLAlchemy Core
- [ ] Implementacja funkcji specyficznych dla każdej bazy
- [ ] Utworzenie narzędzia do migracji danych
- [ ] Napisanie testów dla wszystkich trzech baz
- [ ] Refaktoryzacja istniejącego kodu do używania nowego systemu
- [ ] Dokumentacja i przykłady użycia

## Dodatkowe wskazówki

1. **Kompatybilność wsteczna:**
   - Stwórz wrapper który zachowa istniejące API
   - Stopniowo migruj kod do nowego systemu

2. **Wydajność:**
   - SQLAlchemy Core jest bardzo wydajne dla operacji bulk
   - Używaj `execute_many()` dla wielu insertów
   - Wykorzystuj connection pooling (automatyczny w SQLAlchemy)

3. **Bezpieczeństwo:**
   - Zawsze używaj parametryzowanych zapytań (SQLAlchemy to wymusza)
   - Przechowuj hasła do baz w zmiennych środowiskowych

4. **Monitorowanie:**
   - SQLAlchemy ma wbudowane logowanie zapytań
   - Możesz łatwo dodać metryki wydajności

## Rezultat oczekiwany
- Kod używający SQLAlchemy Core jako warstwy abstrakcji
- Możliwość zmiany silnika bazy danych poprzez zmianę jednej linii konfiguracji
- Zachowanie pełnej funkcjonalności obecnego systemu
- Lepsza testowalność i maintainability
- Wykorzystanie sprawdzonej, dojrzałej biblioteki zamiast własnej implementacji