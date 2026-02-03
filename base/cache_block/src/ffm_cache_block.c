// Implementation of the FFM Cache Blocking / Tiling Utility

#define _GNU_SOURCE
#include "ffm_cache_block.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <math.h>

struct cache_level_entry
{
  int level;         // 1,2,3
  char type[32];     // Data, Instruction, Unified
  size_t size_bytes; // raw size in bytes
};

struct ffm_cache_info
{
  struct cache_level_entry entries[4]; // indexed by level 1..3 (we'll ignore 0)
  int present[4];                      // present flags
};

// Parse strings like "32K" "256K" "3M" into bytes
static size_t parse_size_km(const char *s)
{
  if (!s)
    return 0;
  char *end = NULL;
  errno = 0;
  long val = strtol(s, &end, 10);
  if (end == s)
    return 0;
  while (isspace((unsigned char)*end))
    ++end;
  if (*end == '\0' || *end == 'B')
    return (size_t)val;
  if (*end == 'K' || *end == 'k')
    return (size_t)val * 1024UL;
  if (*end == 'M' || *end == 'm')
    return (size_t)val * 1024UL * 1024UL;
  // fallback: try strtoull whole
  unsigned long long v = strtoull(s, NULL, 10);
  return (size_t)v;
}

// Read a small text file into buffer (caller frees)
static char *read_file_str(const char *path)
{
  FILE *f = fopen(path, "r");
  if (!f)
    return NULL;
  char *line = NULL;
  size_t n = 0;
  ssize_t r = getline(&line, &n, f);
  fclose(f);
  if (r <= 0)
  {
    free(line);
    return NULL;
  }
  // trim newline
  for (ssize_t i = r - 1; i >= 0; --i)
  {
    if (line[i] == '\n' || line[i] == '\r')
      line[i] = '\0';
    else
      break;
  }
  return line;
}

// Probe sysfs for cache info; best-effort
static int probe_sysfs_cache(struct ffm_cache_info *info)
{
  const char *base = "/sys/devices/system/cpu/cpu0/cache";
  DIR *d = opendir(base);
  if (!d)
    return -1;
  struct dirent *ent;
  while ((ent = readdir(d)) != NULL)
  {
    if (ent->d_type != DT_DIR && ent->d_type != DT_LNK && ent->d_type != DT_UNKNOWN)
      continue;
    if (strncmp(ent->d_name, "index", 5) != 0)
      continue;
    char path[512];
    // read level
    snprintf(path, sizeof(path), "%s/%s/level", base, ent->d_name);
    char *level_s = read_file_str(path);
    if (!level_s)
      continue;
    int level = atoi(level_s);
    free(level_s);
    if (level < 1 || level > 3)
      continue;
    // read type
    snprintf(path, sizeof(path), "%s/%s/type", base, ent->d_name);
    char *type_s = read_file_str(path);
    if (!type_s)
      continue;
    // read size
    snprintf(path, sizeof(path), "%s/%s/size", base, ent->d_name);
    char *size_s = read_file_str(path);
    if (!size_s)
    {
      free(type_s);
      continue;
    }
    size_t bytes = parse_size_km(size_s);
    free(size_s);
    // populate
    info->entries[level].level = level;
    strncpy(info->entries[level].type, type_s, sizeof(info->entries[level].type) - 1);
    info->entries[level].type[sizeof(info->entries[level].type) - 1] = '\0';
    info->entries[level].size_bytes = bytes;
    info->present[level] = 1;
    free(type_s);
  }
  closedir(d);
  return 0;
}

// Fallback default cache sizes (common approx values)
static void fill_defaults(struct ffm_cache_info *info)
{
  // L1 data cache: 32KB
  info->entries[1].level = 1;
  strcpy(info->entries[1].type, "Data");
  info->entries[1].size_bytes = 32 * 1024UL;
  info->present[1] = 1;
  // L2: 256KB
  info->entries[2].level = 2;
  strcpy(info->entries[2].type, "Unified");
  info->entries[2].size_bytes = 256 * 1024UL;
  info->present[2] = 1;
  // L3: 8MB
  info->entries[3].level = 3;
  strcpy(info->entries[3].type, "Unified");
  info->entries[3].size_bytes = 8 * 1024UL * 1024UL;
  info->present[3] = 1;
}

ffm_cache_info_t *ffm_cache_init(void)
{
  ffm_cache_info_t *info = calloc(1, sizeof(*info));
  if (!info)
    return NULL;
  // try probe
  if (probe_sysfs_cache(info) != 0)
  {
    // fallback defaults
    fill_defaults(info);
  }
  return info;
}

void ffm_cache_free(ffm_cache_info_t *info)
{
  if (!info)
    return;
  free(info);
}

void ffm_cache_print(ffm_cache_info_t *info)
{
  if (!info)
    return;
  printf("Detected cache levels:\n");
  for (int i = 1; i <= 3; ++i)
  {
    if (info->present[i])
    {
      printf(" L%d: %s - %zu bytes\n", info->entries[i].level, info->entries[i].type, info->entries[i].size_bytes);
    }
    else
    {
      printf(" L%d: not present\n", i);
    }
  }
}

size_t ffm_cache_compute_tile(ffm_cache_info_t *info, int level, size_t elem_size, double occupancy_fraction)
{
  if (!info)
    return 0;
  if (level < 1 || level > 3)
    return 0;
  if (elem_size == 0)
    return 0;
  if (!(occupancy_fraction > 0.0 && occupancy_fraction <= 1.0))
    return 0;
  if (!info->present[level])
    return 0;

  size_t cache_bytes = info->entries[level].size_bytes;
  // use only occupancy_fraction of cache
  double usable = (double)cache_bytes * occupancy_fraction;
  // For matrix multiply, aim to fit three blocks (A_block, B_block, C_block)
  // into the cache: usable / 3 bytes for one block.
  double per_block = usable / 3.0;
  if (per_block < (double)elem_size)
    return 1;
  // compute square tile dimension: tile * tile * elem_size <= per_block
  double tile_d = floor(sqrt(per_block / (double)elem_size));
  if (tile_d < 1.0)
    return 1;
  size_t tile = (size_t)tile_d;
  return tile;
}
