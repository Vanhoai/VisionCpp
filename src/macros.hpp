//
// Created by VanHoai on 30/5/25.
//

#ifndef MACROS_H
#define MACROS_H

#include <iostream>
#include <vector>

#define ms(s, n)      memset(s, n, sizeof(s))
#define all(a)        a.begin(), a.end()
#define sz(a)         int((a).size())
#define FOR(i, a, b)  for (int i = (a); i <= (b); ++i)
#define FORD(i, a, b) for (int i = (a); i >= b; --i)

#define PB push_back
#define MP make_pair
#define F  first
#define S  second

typedef long long ll;
typedef std::pair<int, int> pi;
typedef std::vector<int> vi;
typedef std::vector<pi> vii;
typedef std::vector<vi> vvi;

constexpr int MOD = (int) 1e9 + 7;
constexpr int INF = (int) 1e9 + 1;
constexpr int DEG = (int) 10001;

inline ll gcd(const ll a, const ll b) { return b == 0 ? a : gcd(b, a % b); }
inline ll lcm(const ll a, const ll b) { return a / gcd(a, b) * b; }

#endif   // MACROS_H
