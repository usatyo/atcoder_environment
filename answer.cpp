#include <atcoder/all>
#include <bits/stdc++.h>
using namespace std;
using namespace atcoder;
using msec = chrono::milliseconds;
using Time = chrono::system_clock::time_point;
using ll = long long;
using vi = vector<int>;
using vvi = vector<vi>;

class Timer {
  Time start_time, end_time;

public:
  Timer() { start_time = chrono::system_clock::now(); }
  double time() {
    end_time = chrono::system_clock::now();
    return chrono::duration_cast<msec>(end_time - start_time).count();
  }
};

struct Grid {
  int x, y;
};

int dist(Grid a, Grid b) { return abs(a.x - b.x) + abs(a.y - b.y); }

class Station {
public:
  Grid pos;
  Station(Grid pos) : pos(pos) {}
};

class Plan {
public:
  vvi order;
  Plan() {}
  vvi neighbor() { return order; }
};

class Problem {
  int n, m, k, t;
  vector<Station> stations;
  map<pair<int, int>, int> grid_to_station;
  vvi rails;

public:
  Problem() {
    cin >> n >> m >> k >> t;

    for (int i = -20; i <= 20; i++) {
      for (int j = -20; j <= 20; j++) {
        // 回転対称になるよう調整
        int x = i * 2 + j * 3 + 4;
        int y = -i * 3 + j * 2;
        if (x < 2 || x >= n - 2 || y < 2 || y >= n - 2) {
          continue;
        }
        stations.push_back(Station({x, y}));
        for (int p = -2; p <= 2; p++) {
          for (int q = -2; q <= 2; q++) {
            if (abs(p) + abs(q) <= 2) {
              grid_to_station[{x + p, y + q}] = stations.size();
            }
          }
        }
      }

      for (auto &station : stations) {
        cout << station.pos.x << " " << station.pos.y << endl;
      }

      for (int i = 0; i < m; i++) {
        int sx, sy, tx, ty;
        cin >> sx >> sy >> tx >> ty;
        if (grid_to_station.at({sx, sy}) == 0 ||
            grid_to_station.at({tx, ty}) == 0) {
          continue;
        }
      }
    }
  }
  ll calc_score(vvi order, bool output) { return 0; }
};

class Solver {
  const double TIME_LIMIT = 2900;
  const double start_temp = 1000; // 初期温度
  const double end_temp = -1000;  // 終了温度
  Timer timer = Timer();

public:
  void solve() {
    double start_time = clock();
    Problem problem = Problem();
    Plan plan = Plan();
    while (true) {
      double current_time = timer.time();
      if (current_time > TIME_LIMIT) {
        break;
      }
      double temp =
          start_temp + (end_temp - start_temp) * current_time / TIME_LIMIT;
      for (int i = 0; i < 100; i++) {
        vvi neighbor = plan.neighbor();
        double prob = exp((problem.calc_score(plan.order, false) -
                           problem.calc_score(neighbor, false)) /
                          temp);
        if (prob >
            ((double)rand()) / ((double)RAND_MAX + 1)) { // 確率probで遷移する
          plan.order = neighbor;
        }
      }
    }
  }
};

int main() {
  Solver solver = Solver();
  solver.solve();
  return 0;
}
