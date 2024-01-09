#include <iostream>
#include <vector>
#include <bitset>
#include <map>
#include <climits>
#include <queue>
#include <set>
using namespace std;
using ll = long long;
using pii = pair<int,int>;

vector<ll> dist;
//index wo kanrisuru

vector<int> junban;

class iint{
    public:
    int val;
    bool operator< (iint r) const {
        return dist[val] < dist[r.val];
    }
};

pair<vector<ll>,vector<vector<int>>> dijkstra(vector<vector<pair<ll,int>>> const& g,int start){
    int n = g.size();
    vector<ll> dist(n,LLONG_MAX / 3);
    vector<vector<int>> prev(n);
    priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<pair<ll,int>>> que;
    que.push(pair(0LL,start));
    dist[start] = 0;
    vector<bool> visited(n,false);
    junban.clear();
    while(!que.empty()){
        int fr = que.top().second; que.pop();
        if(visited[fr])continue;
        visited[fr] = true;
        junban.push_back(fr);
        for(int i = 0; i < g[fr].size(); i++){
            ll w;
            int to;
            tie(w,to) = g[fr][i];
            //cout << fr << ' ' << to << endl;
            //cout << w << endl;
            if(dist[to] < dist[fr] + w)continue;
            if(dist[to] == dist[fr] + w){
                prev[to].push_back(fr);
            }
            if(dist[to] > dist[fr] + w){
                dist[to] = dist[fr] + w;
                prev[to].clear();
                prev[to].resize(0);
                prev[to].push_back(fr);
                que.push(pair(dist[to],to));
            }
        }
    }
    return pair(dist,prev);
}


int main(void){
    int n,m;
    cin >> n >> m;
    vector<vector<pair<ll,int>>> g(n),gg(n);
    vector<int> u(m),v(m);
    vector<ll> w(m);
    map<pii,ll> mp;
    for(int i = 0; i < n; i++){
        g[i].clear();
        gg[i].clear();
    }
    for(int i = 0; i < m; i++){
        cin >> u[i] >> v[i] >> w[i];
        u[i]--; v[i]--;
        if(mp.find(pair(u[i],v[i])) == mp.end())mp[pair(u[i],v[i])] = w[i];
        else mp[pair(u[i],v[i])] = min(mp[pair(u[i],v[i])],w[i]);
        g[u[i]].push_back(pair(w[i],v[i]));
        gg[v[i]].push_back(pair(w[i],u[i]));
    }
    vector<vector<int>> prev;
    tie(dist,prev) = dijkstra(g,0);
    set<pii> dijkstra_bridge;

    //ten1 kara gyakuhen mite bridge janaiyatu kesu

    vector<ll> dist2(n,LLONG_MAX / 2);
    vector<vector<int>> prev2(n);
    tie(dist2,prev2) = dijkstra(gg,1);

    for(int i = 0; i < n; i++){
        if(dist[i] + dist2[i] > dist[1])continue;
        for(int j = 0; j < prev[i].size(); j++){
            dijkstra_bridge.insert(pair(i,prev[i][j]));
            dijkstra_bridge.insert(pair(prev[i][j],i));
        }
    }

    vector<bool> visited(n,false);
    for(int i = junban.size() - 1; i >= 0; i--){
        if(visited[i])continue;
        int idx = junban[i - 1];
        if(dist[idx] + dist2[i] > dist[1])continue;
        if(prev[idx].size() >= 2){
            set<iint> st;
            for(int j = 0; j < prev[idx].size(); j++){
                iint temp;
                temp.val = prev[idx][j];
                st.insert(temp);
                dijkstra_bridge.erase(pair(idx,prev[idx][j]));
                dijkstra_bridge.erase(pair(prev[idx][j],idx));
            }

            while(st.size() >= 2){
                iint temp = *st.begin(); st.erase(st.begin());
                idx = temp.val;
                if(visited[idx])continue;
                visited[idx] = true;
                for(int j = 0; j < prev[idx].size(); j++){
                    dijkstra_bridge.erase(pair(idx,prev[idx][j]));
                    dijkstra_bridge.erase(pair(prev[idx][j],idx));
                    temp.val = prev[idx][j];
                    st.insert(temp);
                }
            }
        }
    }



    for(int i = 0; i < m; i++){
        if(dijkstra_bridge.find(pair(u[i],v[i])) != dijkstra_bridge.end() && mp[pair(u[i],v[i])] == w[i]){
            cout << "SAD" << endl;
            continue;
        }
        if(dist[v[i]] + dist2[u[i]] + w[i] < dist[1]){
            cout << "HAPPY" << endl;
            continue;
        }
        cout << "SOSO" << endl;
    }

    return 0;
}
