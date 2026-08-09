#include "database_manager.h"
#include <stddef.h>

namespace DatabaseAbstraction {
namespace {

static_assert(sizeof(uint32_t) == 4U,
              "DatabaseManager requires a 32-bit uint32_t");
static_assert(sizeof(sqlite3_int64) == 8U,
              "DatabaseManager requires a 64-bit sqlite3_int64");
static_assert(sizeof(int) >= sizeof(int32_t),
              "SQLite requires an int type that can hold 32-bit values");
static_assert(sizeof(unsigned int) >= sizeof(uint32_t),
              "Circular-buffer SQL formatting requires 32-bit unsigned int");

const uint32_t SQLITE_INT_MAX_VALUE = 0x7FFFFFFFU;

const char CIRCULAR_TRIGGER_PREFIX[] =
    "__database_manager_circular_buffer_";
const char CIRCULAR_BUFFER_SAVEPOINT_BEGIN[] =
    "SAVEPOINT __database_manager_circular_buffer_setup;";
const char CIRCULAR_BUFFER_SAVEPOINT_ROLLBACK[] =
    "ROLLBACK TO __database_manager_circular_buffer_setup;";
const char CIRCULAR_BUFFER_SAVEPOINT_RELEASE[] =
    "RELEASE __database_manager_circular_buffer_setup;";

uint32_t return_code(ReturnValues value) {
    return static_cast<uint32_t>(value);
}

bool size_to_sqlite_int(size_t value, int& converted) {
    if (value > static_cast<size_t>(SQLITE_INT_MAX_VALUE)) {
        return false;
    }
    converted = static_cast<int>(value);
    return true;
}

bool size_to_uint32(size_t value, uint32_t& converted) {
    if (value > static_cast<size_t>(UINT32_MAX)) {
        return false;
    }
    converted = static_cast<uint32_t>(value);
    return true;
}

bool sqlite_int_to_uint32(int value, uint32_t& converted) {
    if (value < 0) {
        return false;
    }
    converted = static_cast<uint32_t>(value);
    return true;
}

bool sqlite_int64_to_uint32(sqlite3_int64 value, uint32_t& converted) {
    if ((value < static_cast<sqlite3_int64>(0)) ||
        (value > static_cast<sqlite3_int64>(UINT32_MAX))) {
        return false;
    }
    converted = static_cast<uint32_t>(value);
    return true;
}

bool c_strings_equal(const char* left, const char* right) {
    if ((left == 0) || (right == 0)) {
        return false;
    }
    uint32_t index = 0U;
    while ((left[index] != '\0') && (right[index] != '\0')) {
        char left_character = left[index];
        char right_character = right[index];
        if ((left_character >= 'A') && (left_character <= 'Z')) {
            left_character = static_cast<char>(left_character - 'A' + 'a');
        }
        if ((right_character >= 'A') && (right_character <= 'Z')) {
            right_character = static_cast<char>(
                right_character - 'A' + 'a');
        }
        if (left_character != right_character) {
            return false;
        }
        ++index;
    }
    return left[index] == right[index];
}

uint32_t bounded_length(const char* text, uint32_t limit) {
    if (text == 0) {
        return 0U;
    }

    uint32_t length = 0U;
    while ((length < limit) && (text[length] != '\0')) {
        ++length;
    }
    return length;
}

bool append_literal(DbQuery& query, const char* text) {
    if (text == 0) {
        return false;
    }
    const uint32_t length = bounded_length(text, MAX_QUERY_LENGTH + 1U);
    if ((length > MAX_QUERY_LENGTH) ||
        (query.size() + length > MAX_QUERY_LENGTH)) {
        return false;
    }
    query.append(text, length);
    return true;
}

bool append_string(DbQuery& query, const etl::istring& text) {
    if (query.size() + text.size() > MAX_QUERY_LENGTH) {
        return false;
    }
    query.append(text.data(), text.size());
    return true;
}

uint32_t validate_identifier(const etl::istring& identifier) {
    if (identifier.empty()) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }
    if (identifier.size() > DB_MAX_STRING_LENGTH) {
        return return_code(ReturnValues::STRING_TOO_LONG);
    }
    return return_code(ReturnValues::SUCCESS);
}

bool append_identifier(DbQuery& query, const etl::istring& identifier) {
    uint32_t required = 2U;
    for (uint32_t index = 0U; index < identifier.size(); ++index) {
        required += (identifier[index] == '"') ? 2U : 1U;
    }
    if (query.size() + required > MAX_QUERY_LENGTH) {
        return false;
    }

    query.push_back('"');
    for (uint32_t index = 0U; index < identifier.size(); ++index) {
        if (identifier[index] == '"') {
            query.push_back('"');
        }
        query.push_back(identifier[index]);
    }
    query.push_back('"');
    return true;
}

bool append_qualifier(DbQuery& query, const etl::istring& qualifier) {
    if (qualifier.empty()) {
        return true;
    }
    return append_literal(query, " ") && append_string(query, qualifier);
}

bool append_placeholders(DbQuery& query, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        if ((index != 0U) && !append_literal(query, ",")) {
            return false;
        }
        if (!append_literal(query, "?")) {
            return false;
        }
    }
    return true;
}

void append_error_text(DbString& destination, const char* text) {
    if (text == 0) {
        return;
    }
    uint32_t index = 0U;
    while ((text[index] != '\0') &&
           (destination.size() < DB_MAX_STRING_LENGTH)) {
        destination.push_back(text[index]);
        ++index;
    }
}

uint32_t transaction_result(uint32_t result) {
    if ((result == return_code(ReturnValues::EXECUTION_FAILED)) ||
        (result == return_code(ReturnValues::PREPARE_FAILED))) {
        return return_code(ReturnValues::TRANSACTION_FAILED);
    }
    return result;
}

}  // namespace

DbValue::DbValue()
    : type(DbValueType::Null),
      integer_value(static_cast<sqlite3_int64>(0)),
      real_value(0.0),
      string_value(),
      valid(true) {}

DbValue DbValue::null_value() {
    return DbValue();
}

DbValue DbValue::integer(sqlite3_int64 value) {
    DbValue result;
    result.type = DbValueType::Integer;
    result.integer_value = value;
    return result;
}

DbValue DbValue::real(double value) {
    DbValue result;
    result.type = DbValueType::Real;
    result.real_value = value;
    return result;
}

DbValue DbValue::text(const etl::istring& value) {
    if (value.size() > DB_MAX_STRING_LENGTH) {
        DbValue result;
        result.type = DbValueType::Text;
        result.valid = false;
        return result;
    }
    return text(value.data(), static_cast<uint32_t>(value.size()));
}

DbValue DbValue::text(const char* value) {
    if (value == 0) {
        DbValue result;
        result.valid = false;
        return result;
    }
    return text(value, bounded_length(value, DB_MAX_STRING_LENGTH + 1U));
}

DbValue DbValue::text(const char* value, uint32_t length) {
    DbValue result;
    result.type = DbValueType::Text;
    if ((value == 0) || (length > DB_MAX_STRING_LENGTH)) {
        result.valid = false;
        return result;
    }
    result.string_value.assign(value, length);
    return result;
}

DbValue DbValue::blob(const void* value, uint32_t length) {
    DbValue result;
    result.type = DbValueType::Blob;
    if (((value == 0) && (length != 0U)) ||
        (length > DB_MAX_STRING_LENGTH)) {
        result.valid = false;
        return result;
    }
    if (length != 0U) {
        result.string_value.assign(reinterpret_cast<const char*>(value),
                                   length);
    }
    return result;
}

DatabaseManager::DatabaseManager(const etl::istring& dbname,
                                 uint32_t persistent_rows,
                                 uint32_t cols,
                                 OpenMode open_mode)
    : db_(0),
      dbname_(),
      persistent_row_limit_(persistent_rows),
      max_cols_(cols),
      open_mode_(open_mode),
      initialization_status_(return_code(ReturnValues::INVALID_ARGUMENT)),
      schema_version_(0U),
      last_sqlite_error_code_(SQLITE_OK),
      last_error_message_() {
    if ((persistent_rows == 0U) ||
        (cols == 0U) || (cols > DB_MAX_COLS) ||
        ((open_mode != OpenMode::OPEN_OR_CREATE) &&
         (open_mode != OpenMode::OPEN_EXISTING))) {
        return;
    }
    if (dbname.empty()) {
        return;
    }
    if (dbname.size() > DB_MAX_STRING_LENGTH) {
        initialization_status_ = return_code(ReturnValues::STRING_TOO_LONG);
        return;
    }

    dbname_.assign(dbname.data(), dbname.size());
    initialization_status_ = return_code(ReturnValues::SUCCESS);
}

DatabaseManager::~DatabaseManager() {
    if (db_ != 0) {
        sqlite3_close(db_);
        db_ = 0;
    }
}

uint32_t DatabaseManager::open() {
    if (db_ != 0) {
        return return_code(ReturnValues::SUCCESS);
    }
    if (dbname_.empty()) {
        return initialization_status_;
    }

    clear_recorded_error();

    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX;
    if (open_mode_ == OpenMode::OPEN_OR_CREATE) {
        flags |= SQLITE_OPEN_CREATE;
    }
    const int open_code = sqlite3_open_v2(dbname_.c_str(), &db_, flags, 0);
    if (open_code != SQLITE_OK) {
        record_sqlite_error("sqlite3_open_v2", open_code);
        if (db_ != 0) {
            sqlite3_close(db_);
            db_ = 0;
        }
        initialization_status_ = map_sqlite_error(
            open_code, ReturnValues::DATABASE_OPEN_FAILED);
        return initialization_status_;
    }

    schema_version_ = 0U;
    sqlite3_extended_result_codes(db_, 1);
    const char* failed_operation = "PRAGMA journal_mode=DELETE";
    uint32_t result = execute_pragma("PRAGMA journal_mode=DELETE;");
    if (result == return_code(ReturnValues::SUCCESS)) {
        failed_operation = "PRAGMA synchronous=FULL";
        result = execute_pragma("PRAGMA synchronous=FULL;");
    }
    if (result == return_code(ReturnValues::SUCCESS)) {
        failed_operation = "configure circular buffers";
        result = configure_circular_buffers();
    }
    if (result == return_code(ReturnValues::SUCCESS)) {
        failed_operation = "read schema version";
        result = read_schema_version(schema_version_);
    }
    if (result != return_code(ReturnValues::SUCCESS)) {
        record_sqlite_error(failed_operation,
                            sqlite3_extended_errcode(db_));
        sqlite3_close(db_);
        db_ = 0;
    }
    initialization_status_ = result;
    return initialization_status_;
}

uint32_t DatabaseManager::execute_pragma(const char* pragma) {
    const int code = sqlite3_exec(db_, pragma, 0, 0, 0);
    return map_sqlite_error(code, ReturnValues::EXECUTION_FAILED);
}

uint32_t DatabaseManager::read_schema_version(uint32_t& version) const {
    if (db_ == 0) {
        return return_code(ReturnValues::DATABASE_NOT_OPEN);
    }

    sqlite3_stmt* statement = 0;
    int sqlite_code = sqlite3_prepare_v2(
        db_, "PRAGMA main.schema_version;", -1, &statement, 0);
    if (sqlite_code != SQLITE_OK) {
        return map_sqlite_error(sqlite_code, ReturnValues::PREPARE_FAILED);
    }
    if (statement == 0) {
        return return_code(ReturnValues::PREPARE_FAILED);
    }

    sqlite_code = sqlite3_step(statement);
    uint32_t converted_version = 0U;
    uint32_t status = return_code(ReturnValues::SUCCESS);
    if ((sqlite_code != SQLITE_ROW) ||
        !sqlite_int64_to_uint32(sqlite3_column_int64(statement, 0),
                               converted_version)) {
        status = (sqlite_code == SQLITE_ROW)
            ? return_code(ReturnValues::INTERNAL_ERROR)
            : map_sqlite_error(sqlite_code, ReturnValues::EXECUTION_FAILED);
    }

    const int finalize_code = sqlite3_finalize(statement);
    if ((status == return_code(ReturnValues::SUCCESS)) &&
        (finalize_code != SQLITE_OK)) {
        status = map_sqlite_error(finalize_code,
                                  ReturnValues::EXECUTION_FAILED);
    }
    if (status == return_code(ReturnValues::SUCCESS)) {
        version = converted_version;
    }
    return status;
}

uint32_t DatabaseManager::drop_circular_buffer_triggers() {
    sqlite3_stmt* statement = 0;
    int sqlite_code = sqlite3_prepare_v2(
        db_,
        "SELECT name FROM temp.sqlite_master "
        "WHERE type='trigger' "
        "AND name GLOB '__database_manager_circular_buffer_*';",
        -1, &statement, 0);
    if (sqlite_code != SQLITE_OK) {
        return map_sqlite_error(sqlite_code, ReturnValues::PREPARE_FAILED);
    }
    if (statement == 0) {
        return return_code(ReturnValues::PREPARE_FAILED);
    }

    uint32_t status = return_code(ReturnValues::SUCCESS);
    while ((sqlite_code = sqlite3_step(statement)) == SQLITE_ROW) {
        const char* trigger_name = reinterpret_cast<const char*>(
            sqlite3_column_text(statement, 0));
        if (trigger_name == 0) {
            status = return_code(ReturnValues::INTERNAL_ERROR);
            break;
        }

        char* drop_sql = sqlite3_mprintf(
            "DROP TRIGGER temp.\"%w\";", trigger_name);
        if (drop_sql == 0) {
            status = return_code(ReturnValues::OUT_OF_MEMORY);
            break;
        }
        status = execute_pragma(drop_sql);
        sqlite3_free(drop_sql);
        if (status != return_code(ReturnValues::SUCCESS)) {
            break;
        }
    }

    if ((status == return_code(ReturnValues::SUCCESS)) &&
        (sqlite_code != SQLITE_DONE)) {
        status = map_sqlite_error(sqlite_code,
                                  ReturnValues::EXECUTION_FAILED);
    }
    const int finalize_code = sqlite3_finalize(statement);
    if ((status == return_code(ReturnValues::SUCCESS)) &&
        (finalize_code != SQLITE_OK)) {
        status = map_sqlite_error(finalize_code,
                                  ReturnValues::EXECUTION_FAILED);
    }
    return status;
}

uint32_t DatabaseManager::configure_circular_buffer(
    const char* table_name) {
    if (table_name == 0) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }
    const uint32_t table_name_length = bounded_length(
        table_name, DB_MAX_STRING_LENGTH + 1U);
    if (table_name_length == 0U) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }
    if (table_name_length > DB_MAX_STRING_LENGTH) {
        return return_code(ReturnValues::STRING_TOO_LONG);
    }

    static const char* const rowid_aliases[] = {
        "rowid", "_rowid_", "oid"
    };
    bool alias_is_shadowed[3] = {false, false, false};

    char* table_info_sql = sqlite3_mprintf(
        "PRAGMA main.table_info(\"%w\");", table_name);
    if (table_info_sql == 0) {
        return return_code(ReturnValues::OUT_OF_MEMORY);
    }

    sqlite3_stmt* statement = 0;
    int sqlite_code = sqlite3_prepare_v2(
        db_, table_info_sql, -1, &statement, 0);
    sqlite3_free(table_info_sql);
    if (sqlite_code != SQLITE_OK) {
        return map_sqlite_error(sqlite_code, ReturnValues::PREPARE_FAILED);
    }
    if (statement == 0) {
        return return_code(ReturnValues::PREPARE_FAILED);
    }

    while ((sqlite_code = sqlite3_step(statement)) == SQLITE_ROW) {
        const char* column_name = reinterpret_cast<const char*>(
            sqlite3_column_text(statement, 1));
        for (uint32_t alias = 0U; alias < 3U; ++alias) {
            if (c_strings_equal(column_name, rowid_aliases[alias])) {
                alias_is_shadowed[alias] = true;
            }
        }
    }
    uint32_t status = (sqlite_code == SQLITE_DONE)
        ? return_code(ReturnValues::SUCCESS)
        : map_sqlite_error(sqlite_code, ReturnValues::EXECUTION_FAILED);
    const int table_info_finalize_code = sqlite3_finalize(statement);
    if ((status == return_code(ReturnValues::SUCCESS)) &&
        (table_info_finalize_code != SQLITE_OK)) {
        status = map_sqlite_error(table_info_finalize_code,
                                  ReturnValues::EXECUTION_FAILED);
    }
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }

    const char* rowid_alias = 0;
    for (uint32_t alias = 0U; alias < 3U; ++alias) {
        if (alias_is_shadowed[alias]) {
            continue;
        }
        char* probe_sql = sqlite3_mprintf(
            "SELECT %s FROM main.\"%w\" LIMIT 0;",
            rowid_aliases[alias], table_name);
        if (probe_sql == 0) {
            return return_code(ReturnValues::OUT_OF_MEMORY);
        }
        statement = 0;
        sqlite_code = sqlite3_prepare_v2(
            db_, probe_sql, -1, &statement, 0);
        sqlite3_free(probe_sql);
        if (statement != 0) {
            sqlite3_finalize(statement);
        }
        if (sqlite_code == SQLITE_OK) {
            rowid_alias = rowid_aliases[alias];
            break;
        }
    }
    if (rowid_alias == 0) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }

    char* trigger_name = sqlite3_mprintf(
        "%s%s", CIRCULAR_TRIGGER_PREFIX, table_name);
    if (trigger_name == 0) {
        return return_code(ReturnValues::OUT_OF_MEMORY);
    }

    char* trigger_sql = sqlite3_mprintf(
        "CREATE TEMP TRIGGER \"%w\" AFTER INSERT ON main.\"%w\" "
        "WHEN (SELECT COUNT(*) FROM \"%w\") > %u "
        "BEGIN DELETE FROM \"%w\" WHERE %s = "
        "(SELECT %s FROM \"%w\" ORDER BY %s LIMIT 1); END;",
        trigger_name, table_name, table_name,
        static_cast<unsigned int>(persistent_row_limit_), table_name,
        rowid_alias,
        rowid_alias, table_name, rowid_alias);
    sqlite3_free(trigger_name);
    if (trigger_sql == 0) {
        return return_code(ReturnValues::OUT_OF_MEMORY);
    }
    status = execute_pragma(trigger_sql);
    sqlite3_free(trigger_sql);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }

    char* prune_sql = sqlite3_mprintf(
        "DELETE FROM main.\"%w\" WHERE %s IN "
        "(SELECT %s FROM main.\"%w\" ORDER BY %s DESC "
        "LIMIT -1 OFFSET %u);",
        table_name, rowid_alias, rowid_alias, table_name, rowid_alias,
        static_cast<unsigned int>(persistent_row_limit_));
    if (prune_sql == 0) {
        return return_code(ReturnValues::OUT_OF_MEMORY);
    }
    status = execute_pragma(prune_sql);
    sqlite3_free(prune_sql);
    return status;
}

uint32_t DatabaseManager::configure_circular_buffers() {
    if (db_ == 0) {
        return return_code(ReturnValues::DATABASE_NOT_OPEN);
    }

    uint32_t status = execute_pragma(CIRCULAR_BUFFER_SAVEPOINT_BEGIN);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return transaction_result(status);
    }

    status = drop_circular_buffer_triggers();
    sqlite3_stmt* statement = 0;
    int sqlite_code = SQLITE_DONE;
    if (status == return_code(ReturnValues::SUCCESS)) {
        sqlite_code = sqlite3_prepare_v2(
            db_,
            "SELECT name FROM main.sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND sql IS NOT NULL ORDER BY name;",
            -1, &statement, 0);
        if (sqlite_code != SQLITE_OK) {
            status = map_sqlite_error(sqlite_code,
                                      ReturnValues::PREPARE_FAILED);
        } else if (statement == 0) {
            status = return_code(ReturnValues::PREPARE_FAILED);
        }
    }

    if (status == return_code(ReturnValues::SUCCESS)) {
        while ((sqlite_code = sqlite3_step(statement)) == SQLITE_ROW) {
            const char* table_name = reinterpret_cast<const char*>(
                sqlite3_column_text(statement, 0));
            status = configure_circular_buffer(table_name);
            if (status != return_code(ReturnValues::SUCCESS)) {
                break;
            }
        }
        if ((status == return_code(ReturnValues::SUCCESS)) &&
            (sqlite_code != SQLITE_DONE)) {
            status = map_sqlite_error(sqlite_code,
                                      ReturnValues::EXECUTION_FAILED);
        }
    }

    if (statement != 0) {
        const int finalize_code = sqlite3_finalize(statement);
        if ((status == return_code(ReturnValues::SUCCESS)) &&
            (finalize_code != SQLITE_OK)) {
            status = map_sqlite_error(finalize_code,
                                      ReturnValues::EXECUTION_FAILED);
        }
    }

    if (status == return_code(ReturnValues::SUCCESS)) {
        const uint32_t release_status = execute_pragma(
            CIRCULAR_BUFFER_SAVEPOINT_RELEASE);
        return transaction_result(release_status);
    }

    execute_pragma(CIRCULAR_BUFFER_SAVEPOINT_ROLLBACK);
    execute_pragma(CIRCULAR_BUFFER_SAVEPOINT_RELEASE);
    return status;
}

uint32_t DatabaseManager::refresh_circular_buffers() {
    uint32_t current_version = 0U;
    uint32_t status = read_schema_version(current_version);
    if ((status != return_code(ReturnValues::SUCCESS)) ||
        (current_version == schema_version_)) {
        return status;
    }

    status = configure_circular_buffers();
    if (status == return_code(ReturnValues::SUCCESS)) {
        schema_version_ = current_version;
    }
    return status;
}

uint32_t DatabaseManager::initialization_status() const {
    return initialization_status_;
}

bool DatabaseManager::is_open() const {
    return db_ != 0;
}

uint32_t DatabaseManager::close() {
    if (db_ == 0) {
        return return_code(ReturnValues::SUCCESS);
    }
    const int code = sqlite3_close(db_);
    if (code == SQLITE_OK) {
        db_ = 0;
        clear_recorded_error();
        return return_code(ReturnValues::SUCCESS);
    }
    return map_sqlite_error(code, ReturnValues::DATABASE_CLOSE_FAILED);
}

uint32_t DatabaseManager::validate_sql(const etl::istring& sql) const {
    if (sql.empty()) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }
    if (sql.size() > MAX_QUERY_LENGTH) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    int sqlite_length = 0;
    if (!size_to_sqlite_int(sql.size(), sqlite_length)) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    return return_code(ReturnValues::SUCCESS);
}

uint32_t DatabaseManager::bind_values(
    sqlite3_stmt* statement,
    const etl::ivector<DbValue>& bindings) const {
    int binding_count = 0;
    if (!size_to_sqlite_int(bindings.size(), binding_count) ||
        (sqlite3_bind_parameter_count(statement) != binding_count)) {
        return return_code(ReturnValues::BIND_FAILED);
    }

    for (uint32_t index = 0U; index < bindings.size(); ++index) {
        const DbValue& value = bindings[index];
        if (!value.valid) {
            return return_code(ReturnValues::STRING_TOO_LONG);
        }

        const int parameter = static_cast<int>(index + 1U);
        int code = SQLITE_MISUSE;
        switch (value.type) {
            case DbValueType::Null:
                code = sqlite3_bind_null(statement, parameter);
                break;
            case DbValueType::Integer:
                code = sqlite3_bind_int64(statement, parameter,
                                          value.integer_value);
                break;
            case DbValueType::Real:
                code = sqlite3_bind_double(statement, parameter,
                                           value.real_value);
                break;
            case DbValueType::Text: {
                int value_size = 0;
                if (!size_to_sqlite_int(value.string_value.size(),
                                        value_size)) {
                    return return_code(ReturnValues::STRING_TOO_LONG);
                }
                code = sqlite3_bind_text(
                    statement, parameter, value.string_value.data(),
                    value_size,
                    SQLITE_TRANSIENT);
                break;
            }
            case DbValueType::Blob: {
                int value_size = 0;
                if (!size_to_sqlite_int(value.string_value.size(),
                                        value_size)) {
                    return return_code(ReturnValues::STRING_TOO_LONG);
                }
                code = sqlite3_bind_blob(
                    statement, parameter, value.string_value.data(),
                    value_size,
                    SQLITE_TRANSIENT);
                break;
            }
        }
        if (code != SQLITE_OK) {
            return map_sqlite_error(code, ReturnValues::BIND_FAILED);
        }
    }
    return return_code(ReturnValues::SUCCESS);
}

uint32_t DatabaseManager::execute(const etl::istring& sql,
                                  int* changed_rows) {
    DbBindings no_bindings;
    return execute(sql, no_bindings, changed_rows);
}

uint32_t DatabaseManager::execute(
    const etl::istring& sql,
    const etl::ivector<DbValue>& bindings,
    int* changed_rows) {
    uint32_t result = validate_sql(sql);
    if (result != return_code(ReturnValues::SUCCESS)) {
        return result;
    }
    result = open();
    if (result != return_code(ReturnValues::SUCCESS)) {
        return result;
    }
    result = refresh_circular_buffers();
    if (result != return_code(ReturnValues::SUCCESS)) {
        return result;
    }

    sqlite3_stmt* statement = 0;
    int sql_length = 0;
    if (!size_to_sqlite_int(sql.size(), sql_length)) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    int sqlite_code = sqlite3_prepare_v2(
        db_, sql.c_str(), sql_length, &statement, 0);
    if (sqlite_code != SQLITE_OK) {
        return map_sqlite_error(sqlite_code, ReturnValues::PREPARE_FAILED);
    }
    if (statement == 0) {
        return return_code(ReturnValues::PREPARE_FAILED);
    }

    result = bind_values(statement, bindings);
    if (result != return_code(ReturnValues::SUCCESS)) {
        sqlite3_finalize(statement);
        return result;
    }

    do {
        sqlite_code = sqlite3_step(statement);
    } while (sqlite_code == SQLITE_ROW);

    const int finalize_code = sqlite3_finalize(statement);
    result = (sqlite_code == SQLITE_DONE)
        ? map_sqlite_error(finalize_code, ReturnValues::EXECUTION_FAILED)
        : map_sqlite_error(sqlite_code, ReturnValues::EXECUTION_FAILED);
    if ((result == return_code(ReturnValues::SUCCESS)) &&
        (changed_rows != 0)) {
        *changed_rows = sqlite3_changes(db_);
    }
    if (result == return_code(ReturnValues::SUCCESS)) {
        result = refresh_circular_buffers();
    }
    return result;
}

uint32_t DatabaseManager::value_from_column(sqlite3_stmt* statement,
                                            int column_index,
                                            DbValue& value) const {
    switch (sqlite3_column_type(statement, column_index)) {
        case SQLITE_NULL:
            value = DbValue::null_value();
            return return_code(ReturnValues::SUCCESS);
        case SQLITE_INTEGER:
            value = DbValue::integer(sqlite3_column_int64(statement,
                                                          column_index));
            return return_code(ReturnValues::SUCCESS);
        case SQLITE_FLOAT:
            value = DbValue::real(sqlite3_column_double(statement,
                                                        column_index));
            return return_code(ReturnValues::SUCCESS);
        case SQLITE_TEXT: {
            const int sqlite_length = sqlite3_column_bytes(statement,
                                                            column_index);
            uint32_t length = 0U;
            if (!sqlite_int_to_uint32(sqlite_length, length)) {
                return return_code(ReturnValues::INTERNAL_ERROR);
            }
            if (length > DB_MAX_STRING_LENGTH) {
                return return_code(ReturnValues::STRING_TOO_LONG);
            }
            const unsigned char* text = sqlite3_column_text(statement,
                                                            column_index);
            value = DbValue::text(reinterpret_cast<const char*>(text),
                                  length);
            return value.valid ? return_code(ReturnValues::SUCCESS)
                               : return_code(ReturnValues::OUT_OF_MEMORY);
        }
        case SQLITE_BLOB: {
            const int sqlite_length = sqlite3_column_bytes(statement,
                                                            column_index);
            uint32_t length = 0U;
            if (!sqlite_int_to_uint32(sqlite_length, length)) {
                return return_code(ReturnValues::INTERNAL_ERROR);
            }
            if (length > DB_MAX_STRING_LENGTH) {
                return return_code(ReturnValues::STRING_TOO_LONG);
            }
            value = DbValue::blob(sqlite3_column_blob(statement, column_index),
                                  length);
            return value.valid ? return_code(ReturnValues::SUCCESS)
                               : return_code(ReturnValues::OUT_OF_MEMORY);
        }
        default:
            return return_code(ReturnValues::INTERNAL_ERROR);
    }
}

uint32_t DatabaseManager::query(const etl::istring& sql,
                                etl::ivector<DbRow>& result) {
    DbBindings no_bindings;
    return query(sql, no_bindings, result, 0);
}

uint32_t DatabaseManager::query(
    const etl::istring& sql,
    const etl::ivector<DbValue>& bindings,
    etl::ivector<DbRow>& result,
    etl::ivector<DbString>* column_names) {
    result.clear();
    if (column_names != 0) {
        column_names->clear();
    }

    uint32_t status = validate_sql(sql);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }
    status = open();
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }
    status = refresh_circular_buffers();
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }

    sqlite3_stmt* statement = 0;
    int sql_length = 0;
    if (!size_to_sqlite_int(sql.size(), sql_length)) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    int sqlite_code = sqlite3_prepare_v2(
        db_, sql.c_str(), sql_length, &statement, 0);
    if (sqlite_code != SQLITE_OK) {
        return map_sqlite_error(sqlite_code, ReturnValues::PREPARE_FAILED);
    }
    if (statement == 0) {
        return return_code(ReturnValues::PREPARE_FAILED);
    }

    status = bind_values(statement, bindings);
    if (status != return_code(ReturnValues::SUCCESS)) {
        sqlite3_finalize(statement);
        return status;
    }

    const int sqlite_column_count = sqlite3_column_count(statement);
    uint32_t column_count = 0U;
    if (!sqlite_int_to_uint32(sqlite_column_count, column_count)) {
        sqlite3_finalize(statement);
        return return_code(ReturnValues::INTERNAL_ERROR);
    }
    if (column_count > max_cols_) {
        sqlite3_finalize(statement);
        return return_code(ReturnValues::COLUMN_LIMIT_EXCEEDED);
    }

    if (column_names != 0) {
        if (column_names->max_size() < column_count) {
            sqlite3_finalize(statement);
            return return_code(ReturnValues::RESULT_CAPACITY_EXCEEDED);
        }
        for (uint32_t column = 0U; column < column_count; ++column) {
            const int sqlite_column = static_cast<int>(column);
            const char* name = sqlite3_column_name(statement, sqlite_column);
            const uint32_t length = bounded_length(
                name, DB_MAX_STRING_LENGTH + 1U);
            if ((name == 0) || (length > DB_MAX_STRING_LENGTH)) {
                sqlite3_finalize(statement);
                column_names->clear();
                return return_code(ReturnValues::STRING_TOO_LONG);
            }
            DbString stored_name;
            stored_name.assign(name, length);
            column_names->push_back(stored_name);
        }
    }

    while ((sqlite_code = sqlite3_step(statement)) == SQLITE_ROW) {
        if (result.size() >= DB_MAX_ROWS) {
            sqlite3_finalize(statement);
            result.clear();
            if (column_names != 0) {
                column_names->clear();
            }
            return return_code(ReturnValues::ROW_LIMIT_EXCEEDED);
        }
        if (result.full()) {
            sqlite3_finalize(statement);
            result.clear();
            if (column_names != 0) {
                column_names->clear();
            }
            return return_code(ReturnValues::RESULT_CAPACITY_EXCEEDED);
        }

        DbRow row;
        for (uint32_t column = 0U; column < column_count; ++column) {
            DbValue value;
            status = value_from_column(
                statement, static_cast<int>(column), value);
            if (status != return_code(ReturnValues::SUCCESS)) {
                sqlite3_finalize(statement);
                result.clear();
                if (column_names != 0) {
                    column_names->clear();
                }
                return status;
            }
            row.push_back(value);
        }
        result.push_back(row);
    }

    const int finalize_code = sqlite3_finalize(statement);
    status = (sqlite_code == SQLITE_DONE)
        ? map_sqlite_error(finalize_code, ReturnValues::EXECUTION_FAILED)
        : map_sqlite_error(sqlite_code, ReturnValues::EXECUTION_FAILED);
    if (status != return_code(ReturnValues::SUCCESS)) {
        result.clear();
        if (column_names != 0) {
            column_names->clear();
        }
    }
    return status;
}

uint32_t DatabaseManager::create_table(
    const etl::istring& table_name,
    const etl::istring& column_definitions,
    bool if_not_exists) {
    uint32_t status = validate_identifier(table_name);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }
    if (column_definitions.empty()) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }

    DbQuery sql;
    if (!append_literal(sql, "CREATE TABLE ") ||
        (if_not_exists && !append_literal(sql, "IF NOT EXISTS ")) ||
        !append_identifier(sql, table_name) || !append_literal(sql, " (") ||
        !append_string(sql, column_definitions) || !append_literal(sql, ");")) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    return execute(sql);
}

uint32_t DatabaseManager::drop_table(const etl::istring& table_name,
                                     bool if_exists) {
    const uint32_t status = validate_identifier(table_name);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }

    DbQuery sql;
    if (!append_literal(sql, "DROP TABLE ") ||
        (if_exists && !append_literal(sql, "IF EXISTS ")) ||
        !append_identifier(sql, table_name) || !append_literal(sql, ";")) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    return execute(sql);
}

uint32_t DatabaseManager::insert(
    const etl::istring& table_name,
    const etl::istring& column_list,
    const etl::ivector<DbValue>& values,
    sqlite3_int64* inserted_row_id) {
    const uint32_t status = validate_identifier(table_name);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }
    uint32_t value_count = 0U;
    if (!size_to_uint32(values.size(), value_count) ||
        (value_count > max_cols_)) {
        return return_code(ReturnValues::COLUMN_LIMIT_EXCEEDED);
    }

    DbQuery sql;
    if (!append_literal(sql, "INSERT INTO ") ||
        !append_identifier(sql, table_name)) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }

    if (values.empty()) {
        if (!column_list.empty()) {
            return return_code(ReturnValues::INVALID_ARGUMENT);
        }
        if (!append_literal(sql, " DEFAULT VALUES;")) {
            return return_code(ReturnValues::QUERY_TOO_LONG);
        }
    } else {
        if ((!column_list.empty() &&
            (!append_literal(sql, " (") ||
              !append_string(sql, column_list) ||
              !append_literal(sql, ")"))) ||
            !append_literal(sql, " VALUES (") ||
            !append_placeholders(sql, value_count) ||
            !append_literal(sql, ");")) {
            return return_code(ReturnValues::QUERY_TOO_LONG);
        }
    }

    const uint32_t insert_status = execute(sql, values);
    if ((insert_status == return_code(ReturnValues::SUCCESS)) &&
        (inserted_row_id != 0)) {
        *inserted_row_id = sqlite3_last_insert_rowid(db_);
    }
    return insert_status;
}

uint32_t DatabaseManager::select(
    const etl::istring& table_name,
    etl::ivector<DbRow>& result,
    const etl::istring& column_list,
    const etl::istring& qualifier) {
    DbBindings no_bindings;
    return select(table_name, result, column_list, qualifier, no_bindings);
}

uint32_t DatabaseManager::select(
    const etl::istring& table_name,
    etl::ivector<DbRow>& result,
    const etl::istring& column_list,
    const etl::istring& qualifier,
    const etl::ivector<DbValue>& bindings) {
    result.clear();
    const uint32_t status = validate_identifier(table_name);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }

    DbQuery sql;
    if (!append_literal(sql, "SELECT ") ||
        (column_list.empty() ? !append_literal(sql, "*")
                             : !append_string(sql, column_list)) ||
        !append_literal(sql, " FROM ") ||
        !append_identifier(sql, table_name) ||
        !append_qualifier(sql, qualifier) || !append_literal(sql, ";")) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    return query(sql, bindings, result, 0);
}

uint32_t DatabaseManager::update(
    const etl::istring& table_name,
    const etl::istring& assignment_list,
    const etl::istring& qualifier,
    const etl::ivector<DbValue>& bindings,
    int* changed_rows) {
    const uint32_t status = validate_identifier(table_name);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }
    if (assignment_list.empty()) {
        return return_code(ReturnValues::INVALID_ARGUMENT);
    }

    DbQuery sql;
    if (!append_literal(sql, "UPDATE ") ||
        !append_identifier(sql, table_name) ||
        !append_literal(sql, " SET ") ||
        !append_string(sql, assignment_list) ||
        !append_qualifier(sql, qualifier) || !append_literal(sql, ";")) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    return execute(sql, bindings, changed_rows);
}

uint32_t DatabaseManager::delete_rows(
    const etl::istring& table_name,
    const etl::istring& qualifier,
    const etl::ivector<DbValue>& bindings,
    int* changed_rows) {
    const uint32_t status = validate_identifier(table_name);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }

    DbQuery sql;
    if (!append_literal(sql, "DELETE FROM ") ||
        !append_identifier(sql, table_name) ||
        !append_qualifier(sql, qualifier) || !append_literal(sql, ";")) {
        return return_code(ReturnValues::QUERY_TOO_LONG);
    }
    return execute(sql, bindings, changed_rows);
}

uint32_t DatabaseManager::retrieve_column(
    const etl::ivector<DbRow>& result,
    uint32_t column_index,
    etl::ivector<DbValue>& output) const {
    output.clear();
    if (result.size() > DB_MAX_ROWS) {
        return return_code(ReturnValues::ROW_LIMIT_EXCEEDED);
    }
    if (output.max_size() < result.size()) {
        return return_code(ReturnValues::RESULT_CAPACITY_EXCEEDED);
    }
    for (uint32_t row = 0U; row < result.size(); ++row) {
        if (column_index >= result[row].size()) {
            output.clear();
            return return_code(ReturnValues::NOT_FOUND);
        }
        output.push_back(result[row][column_index]);
    }
    return return_code(ReturnValues::SUCCESS);
}

uint32_t DatabaseManager::table_exists(const etl::istring& table_name,
                                       bool& exists) {
    exists = false;
    const uint32_t name_status = validate_identifier(table_name);
    if (name_status != return_code(ReturnValues::SUCCESS)) {
        return name_status;
    }

    DbBindings bindings;
    bindings.push_back(DbValue::text(table_name));
    DbTable result;
    const DbQuery sql(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;");
    const uint32_t status = query(sql, bindings, result, 0);
    if (status == return_code(ReturnValues::SUCCESS)) {
        exists = !result.empty();
    }
    return status;
}

uint32_t DatabaseManager::get_all_table_names(
    etl::ivector<DbString>& names) {
    names.clear();
    DbTable result;
    const DbQuery sql(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name;");
    const uint32_t status = query(sql, result);
    if (status != return_code(ReturnValues::SUCCESS)) {
        return status;
    }
    if (names.max_size() < result.size()) {
        return return_code(ReturnValues::RESULT_CAPACITY_EXCEEDED);
    }
    for (uint32_t index = 0U; index < result.size(); ++index) {
        if (result[index].empty() ||
            (result[index][0].type != DbValueType::Text)) {
            names.clear();
            return return_code(ReturnValues::INTERNAL_ERROR);
        }
        names.push_back(result[index][0].string_value);
    }
    return return_code(ReturnValues::SUCCESS);
}

uint32_t DatabaseManager::begin_transaction() {
    const DbQuery sql("BEGIN TRANSACTION;");
    return transaction_result(execute(sql));
}

uint32_t DatabaseManager::commit_transaction() {
    const DbQuery sql("COMMIT;");
    return transaction_result(execute(sql));
}

uint32_t DatabaseManager::rollback_transaction() {
    const DbQuery sql("ROLLBACK;");
    return transaction_result(execute(sql));
}

sqlite3* DatabaseManager::db_connection_pointer() {
    open();
    return db_;
}

const sqlite3* DatabaseManager::db_connection_pointer() const {
    return db_;
}

sqlite3_int64 DatabaseManager::last_insert_rowid() const {
    return (db_ == 0) ? static_cast<sqlite3_int64>(0)
                      : sqlite3_last_insert_rowid(db_);
}

int DatabaseManager::changes() const {
    return (db_ == 0) ? 0 : sqlite3_changes(db_);
}

int DatabaseManager::last_sqlite_error_code() const {
    return (db_ == 0) ? last_sqlite_error_code_
                      : sqlite3_extended_errcode(db_);
}

const char* DatabaseManager::last_error_message() const {
    if (db_ != 0) {
        return sqlite3_errmsg(db_);
    }
    return last_error_message_.empty()
        ? "database is not open"
        : last_error_message_.c_str();
}

uint32_t DatabaseManager::max_rows() const {
    return persistent_row_limit_;
}

uint32_t DatabaseManager::max_columns() const {
    return max_cols_;
}

void DatabaseManager::clear_recorded_error() {
    last_sqlite_error_code_ = SQLITE_OK;
    last_error_message_.clear();
}

void DatabaseManager::record_sqlite_error(const char* operation,
                                          int sqlite_code) {
    last_sqlite_error_code_ = sqlite_code;
    last_error_message_.clear();
    append_error_text(last_error_message_, operation);
    append_error_text(last_error_message_, ": ");
    const char* message = (db_ == 0) ? sqlite3_errstr(sqlite_code)
                                     : sqlite3_errmsg(db_);
    append_error_text(last_error_message_, message);
}

uint32_t DatabaseManager::map_sqlite_error(int sqlite_code,
                                           ReturnValues fallback) {
    const int primary_code = sqlite_code & 0xFF;
    switch (primary_code) {
        case SQLITE_OK:
        case SQLITE_DONE:
            return return_code(ReturnValues::SUCCESS);
        case SQLITE_BUSY:
            return return_code(ReturnValues::DATABASE_BUSY);
        case SQLITE_LOCKED:
            return return_code(ReturnValues::DATABASE_LOCKED);
        case SQLITE_CONSTRAINT:
            return return_code(ReturnValues::CONSTRAINT_FAILED);
        case SQLITE_NOTFOUND:
            return return_code(ReturnValues::NOT_FOUND);
        case SQLITE_IOERR:
        case SQLITE_CANTOPEN:
            return return_code(ReturnValues::IO_ERROR);
        case SQLITE_NOMEM:
            return return_code(ReturnValues::OUT_OF_MEMORY);
        default:
            return return_code(fallback);
    }
}

}  // namespace DatabaseAbstraction
