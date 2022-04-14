#include <sqlite3.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <tuple>
#include <numeric>
#include <chrono>
#include <cassert>


const char* SELECT_QUERY =
"SELECT lbltype, compid, ncd_formula, seqid_other, seqid_train, ncd_value, seqpart "
"FROM TrainingPairings JOIN Sequences ON seqid_other = seqid "
"WHERE seqpart > 0 "
"ORDER BY lbltype, compid, ncd_formula, seqid_other, seqid_train";


const char* INSERT_QUERY =
"INSERT INTO PairwiseDistances( "
"	lbltype, compid, ncd_formula, dist_aggregator, "
"	seqid_1, seqid_2, dist) "
"VALUES(? , ? , ? , ? , ? , ? , ?)";


int execute(sqlite3_stmt* p_insert_stmt, sqlite3_stmt* p_select_stmt);
int save(sqlite3_stmt* p_insert_stmt, std::vector<std::tuple<int, int, float, int>> rows);


int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cerr << "Missing database filename command line argument." << std::endl;
		return 1;
	}
	if (!std::filesystem::is_regular_file(argv[1]))
	{
		std::cerr << "Cannot find database file " << argv[1] << std::endl;
		return 1;
	}
	
	sqlite3* p_db = nullptr;
	int rc = sqlite3_open(argv[1], &p_db);
	
	if (rc != 0)
	{
		std::cerr << "Error opening database. Error: " << sqlite3_errmsg(p_db) << std::endl;
		return 1;
	}

	sqlite3_exec(p_db, "PRAGMA foreign_keys = ON", nullptr, nullptr, nullptr);
	sqlite3_exec(p_db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

	sqlite3_stmt* p_select_stmt = nullptr;
	rc = sqlite3_prepare_v2(p_db, SELECT_QUERY, -1, &p_select_stmt, nullptr);

	if (rc != SQLITE_OK || p_select_stmt == nullptr)
	{
		std::cerr << "Failed to start select statement. Error: " << sqlite3_errmsg(p_db) << std::endl;
		return 1;
	}

	sqlite3_stmt* p_insert_stmt = nullptr;
	rc = sqlite3_prepare_v2(p_db, INSERT_QUERY, -1, &p_insert_stmt, nullptr);

	if (rc != SQLITE_OK || p_insert_stmt == nullptr)
	{
		std::cerr << "Failed to start insert statement. Error: " << sqlite3_errmsg(p_db) << std::endl;
		return 1;
	}

	rc = execute(p_insert_stmt, p_select_stmt);
	if (rc != SQLITE_OK)
	{
		std::cerr << "Execution failed. Possible error: " << sqlite3_errmsg(p_db) << std::endl;
		return 1;
	}
	sqlite3_finalize(p_insert_stmt);
	sqlite3_finalize(p_select_stmt);

	sqlite3_exec(p_db, "COMMIT", nullptr, nullptr, nullptr);

	sqlite3_close(p_db);
	return 0;
}


int execute(sqlite3_stmt* p_insert_stmt, sqlite3_stmt* p_select_stmt)
{
	size_t count = 0;
	const auto start_time = std::chrono::system_clock::now();

	sqlite3_bind_text(p_insert_stmt, 4, "mp", -1, SQLITE_STATIC);

	int lbltype, compid;
	std::string ncd_formula;
	std::vector<std::tuple<int, int, float, int>> rows;
	bool first_iteration = true;
	bool done = false;
	while (!done)
	{
		int rc = sqlite3_step(p_select_stmt);
		switch (rc)
		{
		case SQLITE_ROW:
			if (first_iteration)
			{
				lbltype = sqlite3_column_int(p_select_stmt, 0);
				compid = sqlite3_column_int(p_select_stmt, 1);
				ncd_formula = (const char*)sqlite3_column_text(p_select_stmt, 2);

				first_iteration = false;

				sqlite3_bind_int(p_insert_stmt, 1, lbltype);
				sqlite3_bind_int(p_insert_stmt, 2, compid);
				sqlite3_bind_text(p_insert_stmt, 3, ncd_formula.c_str(), -1, SQLITE_TRANSIENT);
			}
			// if finished current lbltype/compid/ncd_formula group:
			if (lbltype != sqlite3_column_int(p_select_stmt, 0)
				|| compid != sqlite3_column_int(p_select_stmt, 1)
				|| ncd_formula != (const char*)sqlite3_column_text(p_select_stmt, 2))
			{
				rc = save(p_insert_stmt, std::move(rows));

				lbltype = sqlite3_column_int(p_select_stmt, 0);
				compid = sqlite3_column_int(p_select_stmt, 1);
				ncd_formula = (const char*)sqlite3_column_text(p_select_stmt, 2);

				sqlite3_bind_int(p_insert_stmt, 1, lbltype);
				sqlite3_bind_int(p_insert_stmt, 2, compid);
				sqlite3_bind_text(p_insert_stmt, 3, ncd_formula.c_str(), -1, SQLITE_TRANSIENT);
			}
			// add current row data
			rows.emplace_back(std::make_tuple(
				sqlite3_column_int(p_select_stmt, 3),
				sqlite3_column_int(p_select_stmt, 4),
				(float)sqlite3_column_double(p_select_stmt, 5),
				sqlite3_column_int(p_select_stmt, 6)
			));
			++count;
			if (count % 100 == 0)
			{
				const auto cur_time = std::chrono::system_clock::now();
				const size_t millis =
					std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time).count();
				std::cout << "\rProgress: " << count
					<< " in "
					<< millis
					<< "ms, giving "
					<< (float)millis / (float)count << "ms/seq, or "
					<< (float)(count * 1000) / (float)millis << "seq/s."
					<< std::flush;
			}
			break;
		case SQLITE_DONE:
			done = true;
			break;
		default:
			return rc;
		}
	}

	return SQLITE_OK;
}


int save(sqlite3_stmt* p_insert_stmt, std::vector<std::tuple<int, int, float, int>> rows)
{
	// assume the lbltype/compid/ncd_formula/dist_aggregator are already bound in the statement
	// do a kind of double iteration
	size_t i = 0, i_0 = 0, j = 0;

	// forward j to the point where seqid_other is different
	// to that of i but has the same seqpart
	while (j < rows.size() && (std::get<0>(rows[j]) == std::get<0>(rows[i]) || std::get<3>(rows[i]) != std::get<3>(rows[j])))
		++j;

	float dist = std::numeric_limits<float>::max();
	while (i < rows.size() && j < rows.size())
	{
		if (std::get<1>(rows[i]) != std::get<1>(rows[j]))
		{
			std::cerr << "Oops! Invariant failed." << std::endl;
			return -1;
		}

		// get shortest distance
		if (std::get<2>(rows[i]) + std::get<2>(rows[j]) < dist)
		{
			dist = std::get<2>(rows[i]) + std::get<2>(rows[j]);
		}

		// if end of current seqid_other, need to yield
		if (std::get<0>(rows[j + 1]) != std::get<0>(rows[j]))
		{
			// bind and yield
			sqlite3_bind_double(p_insert_stmt, 7, (double)dist);
			sqlite3_bind_int(p_insert_stmt, 5, std::get<0>(rows[i]));
			sqlite3_bind_int(p_insert_stmt, 6, std::get<0>(rows[j]));
			int rc = sqlite3_step(p_insert_stmt);
			if (rc != SQLITE_DONE)
				return rc;
			sqlite3_reset(p_insert_stmt);
			// do it again but the other way around
			sqlite3_bind_int(p_insert_stmt, 5, std::get<0>(rows[j]));
			sqlite3_bind_int(p_insert_stmt, 6, std::get<0>(rows[i]));
			rc = sqlite3_step(p_insert_stmt);
			if (rc != SQLITE_DONE)
				return rc;
			sqlite3_reset(p_insert_stmt);

			dist = std::numeric_limits<float>::max();
			++j;
			i = i_0;
			// forward j to the point where it has the same seqpart as i
			while (j < rows.size() && std::get<3>(rows[i]) != std::get<3>(rows[j]))
				++j;
		}
		else
		{
			++i;
			++j;
		}

		while (j == rows.size() && i < rows.size())
		{
			// forward i to the point where its seqid_other is different
			while (i + 1 < rows.size() && std::get<0>(rows[i + 1]) == std::get<0>(rows[i]))
				++i;
			++i;

			// reset i_0 and j
			i_0 = i;
			j = i;

			// forward j to the point where seqid_other is different
			// to that of i but has the same seqpart
			while (j < rows.size() && (std::get<0>(rows[j]) == std::get<0>(rows[i]) || std::get<3>(rows[i]) != std::get<3>(rows[j])))
				++j;
		}
	}

	return SQLITE_OK;
}
