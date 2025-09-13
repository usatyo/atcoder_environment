#include <bits/stdc++.h>

#include <atcoder/all>
using namespace std;
using namespace atcoder;

/// @brief 入力データを表す構造体
struct Input {
	/// @brief コンテストの種類 (=26)
	int contest_count;
	/// @brief 合計日数 (=365)
	int day_count;
	/// @brief 減衰係数の配列
	vector<int> minus_factors;
	/// @brief 増加量の配列
	vector<vector<int>> plus_factors;

	/// @brief 入力データを読み込む
	/// @return 読み込んだ入力データ
	static Input read() {
		int day_count, contest_count;
		contest_count = 26;
		cin >> day_count;
		vector<int> minus_factors(26);
		vector<vector<int>> plus_factors(day_count, vector<int>(26));

		for (int i = 0; i < 26; ++i) {
			cin >> minus_factors[i];
		}
		for (int i = 0; i < day_count; ++i) {
			for (int j = 0; j < 26; ++j) {
				cin >> plus_factors[i][j];
			}
		}

		return Input{contest_count, day_count, minus_factors, plus_factors};
	}
};

/// @brief 出力データを表す構造体
struct Output {
	/// @brief 実施するコンテストの配列
	vector<int> contests;

	/// @brief 出力データを構築する
	/// @param orders 選択した注文のリスト
	/// @param route 配達ルート
	Output(vector<int> contests) : contests(contests) {}

	/// @brief 解を出力する
	void print(Input& input) {
		if (contests.size() != input.day_count) {
			throw runtime_error("Invalid contests size");
		}
		for (int i = 0; i < contests.size(); ++i) {
			if (contests[i] < 0 || input.contest_count <= contests[i]) {
				throw runtime_error("Invalid contest number");
			}
			cout << contests[i] + 1 << endl;
		}
	}
};

int calc_minus(const Input& input, const vector<int>& contests, int contest) {
	int last_day = 0;
	int minus = 0;
	for (int day = 0; day < input.day_count; day++) {
		if (contests[day] == contest) {
			last_day = day + 1;
		}
		minus += input.minus_factors[contest] * (day + 1 - last_day);
	}
	return minus;
}

int calc_score(const Input& input, const vector<int>& contests) {
	int score = 0;
	for (int day = 0; day < input.day_count; day++) {
		score += input.plus_factors[day][contests[day]];
	}
	for (int i = 0; i < input.contest_count; i++) {
		score -= calc_minus(input, contests, i);
	}
	return max(1000000 + score, 0);
}

int dif_score(const Input& input, const vector<int>& contests,
			  const vector<int>& minus, int old_contest, int new_contest,
			  int day) {
	int diff = 0;
	diff -= input.plus_factors[day][old_contest];
	diff += input.plus_factors[day][new_contest];
	diff -= calc_minus(input, contests, old_contest) - minus[old_contest];
	diff -= calc_minus(input, contests, new_contest) - minus[new_contest];
	return diff;
}

/// @brief 問題を貪欲法で解く関数
/// @param input 入力データ
/// @return 出力データ
Output solve_greedy(const Input& input) {
	vector<int> contests(input.day_count);
	for (int day = 0; day < input.day_count; day++) {
		int current_contest = 0;
		for (int i = 0; i < input.contest_count; i++) {
			if (input.plus_factors[day][i] >
				input.plus_factors[day][current_contest]) {
				current_contest = i;
			}
		}
		contests[day] = current_contest;
	}
	cerr << "Initial score: " << calc_score(input, contests) << endl;
	return Output(contests);
}

/// @brief
/// 配達先の訪問順序を焼きなまし法で改善する関数（この関数を実装していきます）
/// @param input 入力データ
/// @param output_greedy 貪欲法で求めた出力データ
/// @return 出力データ
Output solve_simulated_annealing(const Input& input,
								 const Output& output_greedy) {
	// 乱数生成器を用意
	// 乱数のシード値は固定のものにしておくと、デバッグがしやすくなります
	mt19937 rand{42};
	uniform_real_distribution<double> zero_one_dist(0.0, 1.0);

	// 焼きなまし法の開始時刻を取得
	auto start_time = chrono::system_clock::now();

	// 制限時間(1.9秒)
	// 2秒ちょうどまでやるとTLEになるので、1.9秒程度にしておくとよい
	const int time_limit = 1900;

	// 開始温度と終了温度
	const double start_temperature = 2e4;
	const double end_temperature = 1e0;

	// 現在の温度
	double current_temperature = start_temperature;

	// 試行回数
	int iteration = 0;

	vector<int> contests = output_greedy.contests;
	vector<int> minus(input.contest_count);
	for (int i = 0; i < input.contest_count; i++) {
		minus[i] = calc_minus(input, contests, i);
	}
	int current_score = calc_score(input, contests);
	int new_score = 0;

	// 焼きなまし法の本体
	while (true) {
		// 現在時刻を取得
		auto current_time = chrono::system_clock::now();

		// 制限時間になったら終了
		if (chrono::duration_cast<chrono::milliseconds>(current_time -
														start_time)
				.count() >= time_limit) {
			break;
		}

		int random_day = rand() % input.day_count;
		int offset = rand() % (input.contest_count - 1) + 1;
		int old_contest = contests[random_day];
		int new_contest = (contests[random_day] + offset) % input.contest_count;
		contests[random_day] = new_contest;
		// new_score = calc_score(input, contests);
		new_score =
			current_score + dif_score(input, contests, minus, old_contest,
									  new_contest, random_day);

		if (new_score > current_score ||
			zero_one_dist(rand) <
				exp((new_score - current_score) / current_temperature)) {
			current_score = new_score;
			minus[old_contest] = calc_minus(input, contests, old_contest);
			minus[new_contest] = calc_minus(input, contests, new_contest);
		} else {
			// 採用されなかったら元に戻す
			contests[random_day] = old_contest;
		}

		// 試行回数のカウントを増やす
		// 進行状況を可視化するため、一定回数ごとに標準エラー出力に出力
		iteration++;
		if (iteration % 100000 == 0) {
			cerr << "iteration: " << iteration
				 << ", current score: " << current_score << endl;
		}

		// 現在の経過時間の割合を計算する
		double progress = (double)chrono::duration_cast<chrono::milliseconds>(
							  current_time - start_time)
							  .count() /
						  (double)time_limit;

		current_temperature = pow(start_temperature, 1.0 - progress) *
							  pow(end_temperature, progress);
	}

	// 試行回数と合計距離を標準エラー出力に出力
	cerr << "--- Result ---" << endl;
	cerr << "iteration     : " << iteration << endl;
	cerr << "curret score: " << current_score << endl;

	return Output(contests);
}

int main() {
	// 入力データを読み込む
	Input input = Input::read();

	// 問題を解く
	Output output_greedy = solve_greedy(input);
	Output output = solve_simulated_annealing(input, output_greedy);

	// 出力する
	output.print(input);

	return 0;
}
