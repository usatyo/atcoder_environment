#include <bits/stdc++.h>

#include <atcoder/all>
using namespace std;
using namespace atcoder;

/// @brief 入力データを表す構造体
struct Input {
	/// @brief 入力データを読み込む
	/// @return 読み込んだ入力データ
	static Input read() { return Input{}; }
};

/// @brief 出力データを表す構造体
struct Output {
	/// @brief 出力データを構築する
	Output() {}

	/// @brief 解を出力する
	void print(Input& input) {}
};

int calc_score(const Input& input) {
	return 0;
}

/// @brief 問題を貪欲法で解く関数
/// @param input 入力データ
/// @return 出力データ
Output solve_greedy(const Input& input) {
	cerr << "Initial score: " << calc_score(input) << endl;
	return Output();
}

/// @brief
/// 配達先の訪問順序を焼きなまし法で改善する関数（この関数を実装していきます）
/// @param input 入力データ
/// @param output_greedy 貪欲法で求めた出力データ
/// @return 出力データ
Output solve_simulated_annealing(const Input& input,
								 const Output& output_greedy) {
	// 乱数生成器を用意
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

	int current_score = calc_score(input);
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

		// 状態の一部をランダムに変更する
		new_score = calc_score(input);

		if (new_score > current_score ||
			zero_one_dist(rand) <
				exp((new_score - current_score) / current_temperature)) {
			current_score = new_score;
		} else {
			// 採用されなかったら元に戻す
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

	return Output();
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
