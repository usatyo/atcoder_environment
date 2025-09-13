/**
 * @file 00_template.cpp
 * @brief AtCoder Heuristic First-step Vol.1のC++向けテンプレートファイル
 */

#include <chrono>
#include <iostream>
#include <vector>

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

/// @brief 問題を解く関数（この関数を実装していきます）
/// @param input 入力データ
/// @return 出力データ
Output solve(const Input& input) {
    // サンプル解法
    // 以下を順に実行するプログラム
    // 1.高橋君は最初オフィスから出発する
    // 2.0番目のレストラン, 0番目の配達先, ..., 49番目のレストラン, 49番目の配達先の順に移動する
    // 3.オフィスに帰る

    std::vector<int> orders;   // 注文の集合
    std::vector<Point> route;  // 配達ルート

    // 1.オフィスからスタート
    route.push_back(input.office);
    Point current_position = input.office;  // 現在地
    int total_dist = 0;                     // 総移動距離

    // 2.レストランと配達先を50箇所（pickup_count）ずつ巡る
    for (int i = 0; i < input.pickup_count; ++i) {
        // 次のレストランに移動する
        // 注文の集合にi番目のレストランを追加
        orders.push_back(i);

        // 配達ルートにi番目のレストランの位置を追加
        route.push_back(input.restaurants[i]);

        // 総移動距離の更新
        total_dist += current_position.dist(input.restaurants[i]);

        // 現在位置をi番目のレストランの位置に更新
        current_position = input.restaurants[i];

        // 次の配達先に移動する
        // 配達ルートにi番目の配達先の位置を追加
        route.push_back(input.destinations[i]);

        // 総移動距離の更新
        total_dist += current_position.dist(input.destinations[i]);

        // 現在位置を最も近い配達先の位置に更新
        current_position = input.destinations[i];
    }

    // 3.オフィスに戻る
    route.push_back(input.office);
    total_dist += current_position.dist(input.office);

    // 合計距離を標準エラー出力に出力
    // 標準エラー出力はデバッグに有効なので、AHCでは積極的に活用していきましょう
    std::cerr << "total distance: " << total_dist << std::endl;

    return Output(orders, route);
}

int main() {
    // 入力データを読み込む
    Input input = Input::read();

    // 問題を解く
    Output output = solve(input);

    // 出力する
    output.print();

    return 0;
}
