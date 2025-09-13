/**
 * @file 03_simulated_annealing.cpp
 * @brief
 * 貪欲法で初期解を求めた後、配達先の訪問順序を焼きなまし法で改善する解法プログラム
 */
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

/// @brief 2次元座標上の点を表す構造体
struct Point {
    int x;
    int y;

    Point(int x, int y) : x(x), y(y) {}

    /// @brief 2点間のマンハッタン距離を計算する
    /// @param p 距離を計算する点
    /// @return 2点間のマンハッタン距離
    int dist(const Point& p) const { return std::abs(x - p.x) + std::abs(y - p.y); }
};

/// @brief 入力データを表す構造体
struct Input {
    /// @brief レストランの数 (=1000)
    int order_count;
    /// @brief 選択する必要のある注文の数 (=50)
    int pickup_count;
    /// @brief AtCoderオフィスの座標 (=(400, 400))
    Point office;
    /// @brief レストランの座標の配列
    std::vector<Point> restaurants;
    /// @brief 目的地の座標の配列
    std::vector<Point> destinations;

    /// @brief 入力データを読み込む
    /// @return 読み込んだ入力データ
    static Input read() {
        int order_count = 1000;
        int pickup_count = 50;
        Point office(400, 400);
        std::vector<Point> restaurants;
        std::vector<Point> destinations;

        for (int i = 0; i < order_count; ++i) {
            int a, b, c, d;
            std::cin >> a >> b >> c >> d;
            restaurants.emplace_back(a, b);
            destinations.emplace_back(c, d);
        }

        return Input{order_count, pickup_count, office, restaurants, destinations};
    }
};

/// @brief 出力データを表す構造体
struct Output {
    /// @brief 移動距離の合計
    int dist_sum;

    /// @brief 選択した注文のリスト
    std::vector<int> orders;

    /// @brief 配達ルート
    std::vector<Point> route;

    /// @brief 出力データを構築する
    /// @param orders 選択した注文のリスト
    /// @param route 配達ルート
    Output(std::vector<int> orders, std::vector<Point> route) : orders(orders), route(route) {
        // 移動距離の合計を計算する
        dist_sum = 0;

        for (int i = 0; i < route.size() - 1; ++i) {
            dist_sum += route[i].dist(route[i + 1]);
        }
    }

    /// @brief 解を出力する
    void print() {
        // 選択した注文の集合を出力する
        std::cout << orders.size();

        for (int i = 0; i < orders.size(); ++i) {
            // 0-indexed -> 1-indexedに変更
            std::cout << " " << orders[i] + 1;
        }

        std::cout << std::endl;

        // 配達ルートを出力する
        std::cout << route.size();

        for (int i = 0; i < route.size(); ++i) {
            std::cout << " " << route[i].x << " " << route[i].y;
        }

        std::cout << std::endl;
    }
};

/// @brief 問題を貪欲法で解く関数
/// @param input 入力データ
/// @return 出力データ
Output solve_greedy(const Input& input) {
    // 貪欲その2
    // 以下を順に実行するプログラム
    // 1.オフィスから距離400以下の注文だけを候補にする
    // 2.高橋君は最初オフィスから出発する
    // 3.訪問したレストランが50軒に達するまで、今いる場所から一番近いレストランに移動することを繰り返す
    // 4.受けた注文を捌ききるまで、今いる場所から一番近い配達先に移動することを繰り返す
    // 5.オフィスに帰る

    std::vector<int> candidates;  // 注文の候補
    std::vector<int> orders;      // 注文の集合
    std::vector<Point> route;     // 配達ルート

    // 1. オフィスから距離400以下の注文だけを候補にする
    for (int i = 0; i < input.order_count; i++) {
        if (input.office.dist(input.restaurants[i]) <= 400 && input.office.dist(input.destinations[i]) <= 400) {
            candidates.push_back(i);
        }
    }

    // 2.オフィスからスタート
    route.push_back(input.office);
    Point current_position = input.office;  // 現在地
    int total_dist = 0;                     // 総移動距離

    // 3.訪問したレストランが50軒に達するまで、今いる場所から一番近いレストランに移動することを繰り返す

    // 同じレストランを2回訪れてはいけないので、訪問済みのレストランを記録する
    std::vector<bool> visited_restaurant(input.order_count, false);

    // pickup_count(=50)回ループ
    for (int i = 0; i < input.pickup_count; ++i) {
        // レストランを全探索して、最も近いレストランを探す
        int nearest_restaurant = 0;  // レストランの番号
        int min_dist = 1000000;      // 最も近いレストランの距離

        // 候補にした注文だけを調べる
        for (int j : candidates) {
            // 既に訪れていたらスキップ
            if (visited_restaurant[j]) {
                continue;
            }

            // 最短距離が更新されたら記録
            int distance = current_position.dist(input.restaurants[j]);

            if (distance < min_dist) {
                min_dist = distance;
                nearest_restaurant = j;
            }
        }

        // 最も近いレストラン(nearest_restaurant)に移動する
        // 現在位置を最も近い配達先の位置に更新
        current_position = input.restaurants[nearest_restaurant];

        // 注文の集合に選んだレストランを追加
        orders.push_back(nearest_restaurant);

        // 配達ルートに現在の位置を追加
        route.push_back(current_position);

        // 訪問済みレストランの配列にtrueをセット
        visited_restaurant[nearest_restaurant] = true;

        // 総移動距離の更新
        total_dist += min_dist;

        // デバッグしやすいよう、標準エラー出力にレストランを出力
        // 標準エラー出力はデバッグに有効なので、AHCでは積極的に活用していきましょう
        Point restaurant_pos = input.restaurants[nearest_restaurant];
        std::cerr << i << "番目のレストラン: p_" << nearest_restaurant << " = (" << restaurant_pos.x << ", "
                  << restaurant_pos.y << ")" << std::endl;
    }

    // 4.受けた注文を捌ききるまで、今いる場所から一番近い配達先に移動することを繰り返す

    // 行かなければいけない配達先のリスト
    // ordersは最終的に出力しなければならないので、ここでコピーを作成しておく
    // 配達先を訪問したらこのリストから1つずつ削除していく
    std::vector<int> destinations(orders);

    // pickup_count(=50)回ループ
    for (int i = 0; i < input.pickup_count; ++i) {
        // 配達先を全探索して、最も近い配達先を探す
        int nearest_index = 0;                                  // 配達先リストのインデックス
        int nearest_destination = destinations[nearest_index];  // 配達先の番号
        int min_dist = 1000000;                                 // 最も近い配達先の距離

        // 0～999まで全探索するのではなく、50個のレストランに対応した配達先を全探索することに注意
        for (int j = 0; j < destinations.size(); ++j) {
            // 最短距離が更新されたら記録
            int distance = current_position.dist(input.destinations[destinations[j]]);

            if (distance < min_dist) {
                min_dist = distance;
                nearest_index = j;
                nearest_destination = destinations[j];
            }
        }

        // 最も近い配達先(nearest_destination)に移動する
        // 現在位置を最も近い配達先の位置に更新
        current_position = input.destinations[nearest_destination];

        // 配達ルートに現在の位置を追加
        route.push_back(current_position);

        // 配達先のリストから削除
        destinations.erase(destinations.begin() + nearest_index);

        // 総移動距離の更新
        total_dist += min_dist;

        // デバッグしやすいよう、標準エラー出力に配達先を出力
        Point destination_pos = input.destinations[nearest_destination];
        std::cerr << i << "番目の配達先: q_" << nearest_destination << " = (" << destination_pos.x << ", "
                  << destination_pos.y << ")" << std::endl;
    }

    // 5.オフィスに戻る
    route.push_back(input.office);
    total_dist += current_position.dist(input.office);

    // 合計距離を標準エラー出力に出力
    std::cerr << "total distance: " << total_dist << std::endl;

    return Output(orders, route);
}

/// @brief 経路の距離を計算する
/// @param route 経路
/// @return 経路の距離
int get_distance(const std::vector<Point>& route) {
    int dist = 0;
    for (int i = 0; i < route.size() - 1; ++i) {
        dist += route[i].dist(route[i + 1]);
    }
    return dist;
}

/// @brief 配達先の訪問順序を焼きなまし法で改善する関数（この関数を実装していきます）
/// @param input 入力データ
/// @param output_greedy 貪欲法で求めた出力データ
/// @return 出力データ
Output solve_simulated_annealing(const Input& input, const Output& output_greedy) {
    // 焼きなまし法
    // 「ある1つの配達先を訪問する順序を、別の場所に入れ替える」操作を繰り返すことで、経路を改善する

    // 貪欲法で求めた解をコピー(これを初期解とする)
    std::vector<int> orders = output_greedy.orders;
    std::vector<Point> route = output_greedy.route;

    // 現在の経路の距離を計算
    int current_dist = get_distance(route);

    // 乱数生成器を用意
    // 乱数のシード値は固定のものにしておくと、デバッグがしやすくなります
    std::mt19937 rand{42};
    std::uniform_real_distribution<double> zero_one_dist(0.0, 1.0);

    // 焼きなまし法の開始時刻を取得
    auto start_time = std::chrono::system_clock::now();

    // 制限時間(1.9秒)
    // 2秒ちょうどまでやるとTLEになるので、1.9秒程度にしておくとよい
    const int time_limit = 1900;

    // 開始温度と終了温度
    const double start_temperature = 2e2;
    const double end_temperature = 1e0;

    // 現在の温度
    double current_temperature = start_temperature;

    // 試行回数
    int iteration = 0;

    // 焼きなまし法の本体
    while (true) {
        // 現在時刻を取得
        auto current_time = std::chrono::system_clock::now();

        // 制限時間になったら終了
        if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() >= time_limit) {
            break;
        }

        // 訪問先が配達先であるようなインデックスの中から、
        // 「i番目の訪問先をj番目に移動」する操作をランダムに選ぶことで、
        // ある配達先を訪れる順序を他の配達先の間に変える
        // 貪欲法で求めた解では、配達先の訪問順序は0-indexedで51番目～100番目であることに注意
        // (AtCoderオフィス、レストラン50軒、配達先50軒、AtCoderオフィスの順に並んでいる)

        // 訪問先が配達先であるようなインデックスの中から i, j をランダムに選ぶ
        int i = rand() % input.pickup_count + input.pickup_count + 1;
        int j = rand() % input.pickup_count + input.pickup_count + 1;

        // i番目の訪問先をj番目に移動する操作を行う
        Point point_to_move = route[i];
        route.erase(route.begin() + i);
        route.insert(route.begin() + j, point_to_move);

        // 操作後の経路の距離を計算
        int new_dist = get_distance(route);

        // 【穴埋め】操作後の距離が操作前以下なら採用する
        // 【穴埋め】操作前より悪化していても、確率で採用する(悪化度合いが小さく、温度が高いほど採用されやすい)
        // 【ヒント】採用確率(0.0以上1.0未満)は std::exp((current_dist - new_dist) / current_temperature) で計算できる
        // 【ヒント】zero_one_dist(rand) と書くと、0.0以上1.0未満の乱数が得られる
        if (current_dist /* put your code here */) {
            current_dist = new_dist;
        } else {
            // 採用されなかったら元に戻す
            route.erase(route.begin() + j);
            route.insert(route.begin() + i, point_to_move);
        }

        // 試行回数のカウントを増やす
        // 進行状況を可視化するため、一定回数ごとに現在の試行回数と合計距離を標準エラー出力に出力
        iteration++;
        if (iteration % 100000 == 0) {
            std::cerr << "iteration: " << iteration << ", total distance: " << current_dist << std::endl;
        }

        // 現在の経過時間の割合を計算する
        double progress = (double) std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() / (double) time_limit;
        // 【穴埋め】温度の更新
        // 【ヒント】現在の経過時間の割合に対する温度は std::pow(start_temperature, 1.0 - progress) * std::pow(end_temperature, progress) で計算できる
        /* put your code here */

        // ここまで穴埋めして実行できるようになったら、
        // 開始温度(start_temperature)と終了温度(end_temperature)を変えてみて、実行結果がどう変わるかを確認してみましょう
    }

    // 試行回数と合計距離を標準エラー出力に出力
    std::cerr << "--- Result ---" << std::endl;
    std::cerr << "iteration     : " << iteration << std::endl;
    std::cerr << "total distance: " << current_dist << std::endl;

    return Output(orders, route);
}

int main() {
    // 入力データを読み込む
    Input input = Input::read();

    // 問題を解く
    Output output_greedy = solve_greedy(input);
    Output output = solve_simulated_annealing(input, output_greedy);

    // 出力する
    output.print();

    return 0;
}