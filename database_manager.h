#ifndef DATABASE_MANAGER_DATABASE_MANAGER_H
#define DATABASE_MANAGER_DATABASE_MANAGER_H

#ifndef MAX_QUERY_LENGTH
#define MAX_QUERY_LENGTH 1024U
#endif

#ifndef DB_MAX_STRING_LENGTH
#define DB_MAX_STRING_LENGTH 256U
#endif

#ifndef DB_MAX_COLS
#define DB_MAX_COLS 32U
#endif

#ifndef DB_MAX_ROWS
#define DB_MAX_ROWS 128U
#endif

#include <stdint.h>

#include <etl/string.h>
#include <etl/vector.h>
#include <sqlite3.h>

namespace DatabaseAbstraction {

enum class ReturnValues : uint32_t {
    SUCCESS = 0U,
    INVALID_ARGUMENT,
    DATABASE_NOT_OPEN,
    DATABASE_OPEN_FAILED,
    DATABASE_CLOSE_FAILED,
    QUERY_TOO_LONG,
    STRING_TOO_LONG,
    COLUMN_LIMIT_EXCEEDED,
    ROW_LIMIT_EXCEEDED,
    RESULT_CAPACITY_EXCEEDED,
    PREPARE_FAILED,
    BIND_FAILED,
    EXECUTION_FAILED,
    DATABASE_BUSY,
    DATABASE_LOCKED,
    CONSTRAINT_FAILED,
    NOT_FOUND,
    TRANSACTION_FAILED,
    IO_ERROR,
    OUT_OF_MEMORY,
    INTERNAL_ERROR
};

enum class DbValueType : uint32_t {
    Null = 0U,
    Integer,
    Real,
    Text,
    Blob
};

enum class OpenMode : uint32_t {
    OPEN_OR_CREATE = 0U,
    OPEN_EXISTING
};

typedef etl::string<DB_MAX_STRING_LENGTH> DbString;
typedef etl::string<MAX_QUERY_LENGTH> DbQuery;

struct DbValue {
    DbValueType type;
    sqlite3_int64 integer_value;
    double real_value;
    DbString string_value;
    bool valid;

    DbValue();

    static DbValue null_value();
    static DbValue integer(sqlite3_int64 value);
    static DbValue real(double value);
    static DbValue text(const etl::istring& value);
    static DbValue text(const char* value);
    static DbValue text(const char* value, uint32_t length);
    static DbValue blob(const void* value, uint32_t length);
};

typedef etl::vector<DbValue, DB_MAX_COLS> DbRow;
typedef etl::vector<DbRow, DB_MAX_ROWS> DbTable;
typedef etl::vector<DbValue, DB_MAX_COLS> DbBindings;
typedef etl::vector<DbValue, DB_MAX_ROWS> DbColumn;
typedef etl::vector<DbString, DB_MAX_COLS> DbColumnNames;
typedef etl::vector<DbString, DB_MAX_ROWS> DbTableNames;

class DatabaseManager {
public:
    explicit DatabaseManager(const etl::istring& dbname,
                             uint32_t persistent_rows = DB_MAX_ROWS,
                             uint32_t cols = DB_MAX_COLS,
                             OpenMode open_mode =
                                 OpenMode::OPEN_OR_CREATE);
    ~DatabaseManager();

    DatabaseManager(const DatabaseManager&) = delete;
    DatabaseManager& operator=(const DatabaseManager&) = delete;

    uint32_t initialization_status() const;
    bool is_open() const;
    uint32_t open();
    uint32_t close();

    uint32_t execute(const etl::istring& sql,
                     int* changed_rows = 0);
    uint32_t execute(const etl::istring& sql,
                     const etl::ivector<DbValue>& bindings,
                     int* changed_rows = 0);

    uint32_t query(const etl::istring& sql,
                   etl::ivector<DbRow>& result);
    uint32_t query(const etl::istring& sql,
                   const etl::ivector<DbValue>& bindings,
                   etl::ivector<DbRow>& result,
                   etl::ivector<DbString>* column_names = 0);

    uint32_t create_table(const etl::istring& table_name,
                          const etl::istring& column_definitions,
                          bool if_not_exists = true);
    uint32_t drop_table(const etl::istring& table_name,
                        bool if_exists = true);

    uint32_t insert(const etl::istring& table_name,
                    const etl::istring& column_list,
                    const etl::ivector<DbValue>& values,
                    sqlite3_int64* inserted_row_id = 0);

    uint32_t select(const etl::istring& table_name,
                    etl::ivector<DbRow>& result,
                    const etl::istring& column_list,
                    const etl::istring& qualifier);
    uint32_t select(const etl::istring& table_name,
                    etl::ivector<DbRow>& result,
                    const etl::istring& column_list,
                    const etl::istring& qualifier,
                    const etl::ivector<DbValue>& bindings);

    uint32_t update(const etl::istring& table_name,
                    const etl::istring& assignment_list,
                    const etl::istring& qualifier,
                    const etl::ivector<DbValue>& bindings,
                    int* changed_rows = 0);
    uint32_t delete_rows(const etl::istring& table_name,
                         const etl::istring& qualifier,
                         const etl::ivector<DbValue>& bindings,
                         int* changed_rows = 0);

    uint32_t retrieve_column(const etl::ivector<DbRow>& result,
                             uint32_t column_index,
                             etl::ivector<DbValue>& output) const;

    uint32_t table_exists(const etl::istring& table_name, bool& exists);
    uint32_t get_all_table_names(etl::ivector<DbString>& names);

    uint32_t begin_transaction();
    uint32_t commit_transaction();
    uint32_t rollback_transaction();

    sqlite3* db_connection_pointer();
    const sqlite3* db_connection_pointer() const;
    sqlite3_int64 last_insert_rowid() const;
    int changes() const;
    int last_sqlite_error_code() const;
    const char* last_error_message() const;

    uint32_t max_rows() const;
    uint32_t max_columns() const;

private:
    sqlite3* db_;
    DbString dbname_;
    uint32_t persistent_row_limit_;
    uint32_t max_cols_;
    OpenMode open_mode_;
    uint32_t initialization_status_;
    uint32_t schema_version_;
    int last_sqlite_error_code_;
    DbString last_error_message_;

    uint32_t execute_pragma(const char* pragma);
    uint32_t read_schema_version(uint32_t& version) const;
    uint32_t refresh_circular_buffers();
    uint32_t configure_circular_buffers();
    uint32_t configure_circular_buffer(const char* table_name);
    uint32_t drop_circular_buffer_triggers();
    uint32_t validate_sql(const etl::istring& sql) const;
    uint32_t bind_values(sqlite3_stmt* statement,
                         const etl::ivector<DbValue>& bindings) const;
    uint32_t value_from_column(sqlite3_stmt* statement,
                               int column_index,
                               DbValue& value) const;
    void clear_recorded_error();
    void record_sqlite_error(const char* operation, int sqlite_code);
    static uint32_t map_sqlite_error(int sqlite_code,
                                     ReturnValues fallback);
};

}  // namespace DatabaseAbstraction

#endif  // DATABASE_MANAGER_DATABASE_MANAGER_H
