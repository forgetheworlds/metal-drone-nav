#include "navigation_tasks.hpp"
#include <array>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <numeric>
struct Stats { uint64_t attempts=0; uint64_t accepted=0; uint64_t direct=0; uint64_t detour=0; std::array<uint64_t,4> scene_attempts{}; std::array<uint64_t,4> scene_accepts{}; std::array<uint64_t,9> reject{}; };
int scene_index(uint f){return f==1?0:(f==2?1:(f==4?2:3));}
void probe(uint seed,const NavigationTaskConfig& cfg,Stats& st){
 uint rng=seed; WWorld world{};
 for(uint attempt=0;attempt<cfg.max_generation_attempts;attempt++){
  st.attempts++;
  uint scene_seed=wrng(rng);uint sel=wrng(rng)%4u;uint fam=sel==0?1u:(sel==1?2u:(sel==2?4u:5u));int fi=scene_index(fam);st.scene_attempts[fi]++;
  wgenerate(world,scene_seed,fam,cfg.scene_distance_m);
  WVec direction=navigation_task_random_direction(rng);float distance=navigation_task_sample_goal_distance(rng,cfg);WVec start=navigation_task_random_point(rng,cfg.start_min_xyz,cfg.start_max_xyz);WVec goal=wa(start,wm(direction,distance));
  bool goal_in_bounds=navigation_task_inside_bounds(goal,cfg.goal_min_xyz,cfg.goal_max_xyz);
  for(uint d=1;!goal_in_bounds&&d<NAV_TASK_MAX_DIRECTION_ATTEMPTS;d++){direction=navigation_task_random_direction(rng);goal=wa(start,wm(direction,distance));goal_in_bounds=navigation_task_inside_bounds(goal,cfg.goal_min_xyz,cfg.goal_max_xyz);}
  if(!goal_in_bounds){st.reject[1]++;continue;}
  if(!navigation_task_finite(goal.x)||!navigation_task_finite(goal.y)||!navigation_task_finite(goal.z)){st.reject[0]++;continue;}
  if(!navigation_task_inside_bounds(start,cfg.start_min_xyz,cfg.start_max_xyz)||!navigation_task_inside_bounds(goal,cfg.goal_min_xyz,cfg.goal_max_xyz)){st.reject[1]++;continue;}
  float sc=wclearance(world,start,0),gc=wclearance(world,goal,0);
  if(sc<=cfg.minimum_endpoint_clearance_m){st.reject[2]++;continue;}
  if(gc<=cfg.minimum_endpoint_clearance_m){st.reject[3]++;continue;}
  float actual=wl(ws(goal,start));
  if(actual<cfg.goal_distance_min_m||actual>cfg.goal_distance_max_m){st.reject[4]++;continue;}
  NavigationTaskWitness witness{};float direct=0;
  if(!navigation_task_find_witness(world,start,goal,cfg,witness,direct)){st.reject[5]++;continue;}
  if(direct>cfg.minimum_witness_clearance_m)st.direct++;else st.detour++;
  st.accepted++;st.scene_accepts[fi]++;return;
 }
}
int main(){NavigationTaskConfig cfg{};if(!navigation_task_default_config(cfg,NAV_TASK_STAGE_CLUTTER_GOAL,12,.5f))return 1;
 cfg.initial_speed_max_mps=1;Stats s{};std::array<uint64_t,4> actual_family_accepts{};const uint N=20000;uint failures=0,actual_failures=0,mismatches=0;uint firstfail=0;uint64_t totalattempts=0,maxattempts=0;
 for(uint i=1;i<=N;i++){
  Stats one{};probe(i,cfg,one);totalattempts+=one.attempts;maxattempts=std::max(maxattempts,one.attempts);if(one.accepted==0){failures++;if(!firstfail)firstfail=i;}s.attempts+=one.attempts;s.accepted+=one.accepted;
  WWorld actual_world{};NavigationTaskState actual{};uint actual_rng=i;navigation_generate_task(actual_world,actual,actual_rng,cfg);
  if((actual.valid!=0)!=(one.accepted!=0)||actual.generation_attempts!=one.attempts)mismatches++;
  if(actual.valid==0)actual_failures++;
  if(actual.valid){int fi=scene_index(actual.family);actual_family_accepts[fi]++;if(actual.direct_segment_clearance_m>cfg.minimum_witness_clearance_m)s.direct++;else s.detour++;}
  for(int j=0;j<4;j++){s.scene_attempts[j]+=one.scene_attempts[j];s.scene_accepts[j]+=one.scene_accepts[j];}for(int j=0;j<9;j++)s.reject[j]+=one.reject[j];
 }
 std::printf("seeds=%u failures=%u actual_failures=%u mismatches=%u first_failure=%u attempts_total=%llu mean_attempts=%.5f max_attempts=%llu\n",N,failures,actual_failures,mismatches,firstfail,(unsigned long long)totalattempts,double(totalattempts)/N,(unsigned long long)maxattempts);
 std::printf("accepted_direct=%llu accepted_detour=%llu detour_share=%.5f\n",(unsigned long long)s.direct,(unsigned long long)s.detour,double(s.detour)/std::max<uint64_t>(1,s.accepted));
 const char* names[]={"family1 AABB","family2 cylinder","family4 doorway","family5 table/counter"};for(int j=0;j<4;j++)std::printf("%s attempted=%llu accepted=%llu accept_rate=%.5f\n",names[j],(unsigned long long)s.scene_attempts[j],(unsigned long long)actual_family_accepts[j],double(actual_family_accepts[j])/std::max<uint64_t>(1,s.scene_attempts[j]));
 const char* rnames[]={"nonfinite","bounds","start clearance","goal clearance","distance","no witness"};for(int j=0;j<6;j++)std::printf("reject_%s=%llu\n",rnames[j],(unsigned long long)s.reject[j]);
 uint base_seed=43;for(uint episode=0;episode<300;episode++){uint seed=base_seed^((7u+1u)*0x27d4eb2du)^((episode+1u)*0x165667b1u);Stats one{};probe(seed,cfg,one);if(!one.accepted){printf("env7_failure_seed=%u episode_counter=%u episode_number=%u\n",seed,episode,episode+1);break;}}
 Stats failed{};probe(825501529u,cfg,failed);printf("target_failure attempts=%llu accepted=%llu direct=%llu detour=%llu\n",(unsigned long long)failed.attempts,(unsigned long long)failed.accepted,(unsigned long long)failed.direct,(unsigned long long)failed.detour);for(int j=0;j<6;j++)printf("target_reject_%s=%llu\n",rnames[j],(unsigned long long)failed.reject[j]);for(int j=0;j<4;j++)printf("target_family%d_attempts=%llu\n",j,(unsigned long long)failed.scene_attempts[j]);
}
