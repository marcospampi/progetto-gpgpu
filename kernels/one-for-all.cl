
#ifndef WORKGROUP_SIZE
  #define WORKGROUP_SIZE 0
#endif
#ifndef INPUT_ROW_SIZE
  #define INPUT_ROW_SIZE 0
  #define INPUT_VIRTUAL_SIZE 1
#endif

kernel void oneforall(
  global int *g_input,
  global int *g_output,
  global int *g_lengths,
  local  int *l_scratch_a,
  local  int *l_scratch_b,
  global int *g_scratch
) {
  const int workgroup_id = get_group_id(0);
  const int local_id = get_local_id(1);

  prefetch( g_input + workgroup_id * INPUT_ROW_SIZE, INPUT_ROW_SIZE);
  
}