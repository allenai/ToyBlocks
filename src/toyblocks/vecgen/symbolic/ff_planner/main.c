

/*********************************************************************
 * (C) Copyright 2001 Albert Ludwigs University Freiburg
 *     Institute of Computer Science
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 *********************************************************************/


/*
 * THIS SOURCE CODE IS SUPPLIED  ``AS IS'' WITHOUT WARRANTY OF ANY KIND, 
 * AND ITS AUTHOR AND THE JOURNAL OF ARTIFICIAL INTELLIGENCE RESEARCH 
 * (JAIR) AND JAIR'S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL 
 * WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND
 * ANY WARRANTIES OR NON INFRINGEMENT.  THE USER ASSUMES ALL LIABILITY AND
 * RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE AUTHOR NOR
 * JAIR, NOR JAIR'S PUBLISHERS AND DISTRIBUTORS, WILL BE LIABLE FOR 
 * DAMAGES OF ANY KIND RESULTING FROM ITS USE.  Without limiting the 
 * generality of the foregoing, neither the author, nor JAIR, nor JAIR's
 * publishers and distributors, warrant that the Source Code will be 
 * error-free, will operate without interruption, or will meet the needs 
 * of the user.
 */






/*********************************************************************
 * File: main.c
 * Description: The main routine for the FastForward Planner.
 *
 * Author: Joerg Hoffmann 2000
 * 
 *********************************************************************/ 

/** Minor modification by siddharth**/







#include "ff.h"

#include "memory.h"
#include "output.h"

#include "parse.h"

#include "inst_pre.h"
#include "inst_easy.h"
#include "inst_hard.h"
#include "inst_final.h"

#include "orderings.h"

#include "relax.h"
#include "search.h"











/*
 *  ----------------------------- GLOBAL VARIABLES ----------------------------
 */












/*******************
 * GENERAL HELPERS *
 *******************/








/* used to time the different stages of the planner
 */
float gtempl_time = 0, greach_time = 0, grelev_time = 0, gconn_time = 0;
float gsearch_time = 0;


/* the command line inputs
 */
struct _command_line gcmd_line;

/* number of states that got heuristically evaluated
 */
int gevaluated_states = 0;

/* maximal depth of breadth first search
 */
int gmax_search_depth = 0;





/***********
 * PARSING *
 ***********/







/* used for pddl parsing, flex only allows global variables
 */
/* int gbracket_count; */
char *gproblem_name;

/* The current input line number
 */
int lineno = 1;

/* The current input filename
 */
char *gact_filename;

/* The pddl domain name
 */
char *gdomain_name = NULL;

/* loaded, uninstantiated operators
 */
PlOperator *gloaded_ops = NULL;

/* stores initials as fact_list 
 */
PlNode *gorig_initial_facts = NULL;

/* not yet preprocessed goal facts
 */
PlNode *gorig_goal_facts = NULL;

/* axioms as in UCPOP before being changed to ops
 */
PlOperator *gloaded_axioms = NULL;

/* the types, as defined in the domain file
 */
TypedList *gparse_types = NULL;

/* the constants, as defined in domain file
 */
TypedList *gparse_constants = NULL;

/* the predicates and their arg types, as defined in the domain file
 */
TypedListList *gparse_predicates = NULL;

/* the objects, declared in the problem file
 */
TypedList *gparse_objects = NULL;


/* connection to instantiation ( except ops, goal, initial )
 */

/* all typed objects 
 */
FactList *gorig_constant_list = NULL;

/* the predicates and their types
 */
FactList *gpredicates_and_types = NULL;












/*****************
 * INSTANTIATING *
 *****************/









/* global arrays of constant names,
 *               type names (with their constants),
 *               predicate names,
 *               predicate aritys,
 *               defined types of predicate args
 */
Token gconstants[MAX_CONSTANTS];
int gnum_constants = 0;
Token gtype_names[MAX_TYPES];
int gtype_consts[MAX_TYPES][MAX_TYPE];
Bool gis_member[MAX_CONSTANTS][MAX_TYPES];
int gmember_nr[MAX_CONSTANTS][MAX_TYPES];/* nr of object within a type */
int gtype_size[MAX_TYPES];
int gnum_types = 0;
Token gpredicates[MAX_PREDICATES];
int garity[MAX_PREDICATES];
int gpredicates_args_type[MAX_PREDICATES][MAX_ARITY];
int gnum_predicates = 0;





/* the domain in integer (Fact) representation
 */
Operator_pointer goperators[MAX_OPERATORS];
int gnum_operators = 0;
Fact *gfull_initial;
int gnum_full_initial = 0;
WffNode *ggoal = NULL;




/* stores inertia - information: is any occurence of the predicate
 * added / deleted in the uninstantiated ops ?
 */
Bool gis_added[MAX_PREDICATES];
Bool gis_deleted[MAX_PREDICATES];



/* splitted initial state:
 * initial non static facts,
 * initial static facts, divided into predicates
 * (will be two dimensional array, allocated directly before need)
 */
Facts *ginitial = NULL;
int gnum_initial = 0;
Fact **ginitial_predicate;
int *gnum_initial_predicate;



/* the type numbers corresponding to any unary inertia
 */
int gtype_to_predicate[MAX_PREDICATES];
int gpredicate_to_type[MAX_TYPES];

/* (ordered) numbers of types that new type is intersection of
 */
TypeArray gintersected_types[MAX_TYPES];
int gnum_intersected_types[MAX_TYPES];



/* splitted domain: hard n easy ops
 */
Operator_pointer *ghard_operators;
int gnum_hard_operators;
NormOperator_pointer *geasy_operators;
int gnum_easy_operators;



/* so called Templates for easy ops: possible inertia constrained
 * instantiation constants
 */
EasyTemplate *geasy_templates;
int gnum_easy_templates;



/* first step for hard ops: create mixed operators, with conjunctive
 * precondition and arbitrary effects
 */
MixedOperator *ghard_mixed_operators;
int gnum_hard_mixed_operators;



/* hard ''templates'' : pseudo actions
 */
PseudoAction_pointer *ghard_templates;
int gnum_hard_templates;



/* store the final "relevant facts"
 */
Fact grelevant_facts[MAX_RELEVANT_FACTS];
int gnum_relevant_facts = 0;
int gnum_pp_facts = 0;



/* the final actions and problem representation
 */
Action *gactions;
int gnum_actions;
State ginitial_state;
State ggoal_state;









/**********************
 * CONNECTIVITY GRAPH *
 **********************/







/* one ops (actions) array ...
 */
OpConn *gop_conn;
int gnum_op_conn;



/* one effects array ...
 */
EfConn *gef_conn;
int gnum_ef_conn;



/* one facts array.
 */
FtConn *gft_conn;
int gnum_ft_conn;









/*******************
 * SEARCHING NEEDS *
 *******************/








/* the goal state, divided into subsets
 */
State *ggoal_agenda;
int gnum_goal_agenda;



/* byproduct of fixpoint: applicable actions
 */
int *gA;
int gnum_A;



/* communication from extract 1.P. to search engines:
 * 1P action choice
 */
int *gH;
int gnum_H;



/* the effects that are considered true in relaxed plan
 */
int *gin_plan_E;
int gnum_in_plan_E;



/* always stores (current) serial plan
 */
int gplan_ops[MAX_PLAN_LENGTH];
int gnum_plan_ops = 0;



/* stores the states that the current plan goes through
 * ( for knowing where new agenda entry starts from )
 */
State gplan_states[MAX_PLAN_LENGTH + 1];


Plan *gfound_plans = NULL;
int gnum_found_plans = 0;
int gmax_found_plans = 0;

Plan *gdisallowed_plans = NULL;
int gnum_disallowed_plans = 0;
int gmax_disallowed_plans = 0;


/*
 * Helper function to convert action index to string representation
 * Format: (action_name param1 param2 ...)
 */
void action_index_to_string( int index, char *buf, int buf_size )
{
  int i;
  Action *a = gop_conn[index].action;
  int pos = 0;
  
  if ( !a->norm_operator && !a->pseudo_action ) {
    sprintf( buf, "(REACH-GOAL)" );
    return;
  }
  
  pos = sprintf( buf, "(%s", a->name );
  for ( i = 0; i < a->num_name_vars && pos < buf_size - 1; i++ ) {
    pos += sprintf( buf + pos, " %s", gconstants[a->name_inst_table[i]] );
  }
  if ( pos < buf_size - 1 ) {
    sprintf( buf + pos, ")" );
  }
}


/*
 * Helper function to convert a plan to string representation
 * Format: one action per line
 */
void plan_to_string( Plan *plan, char *buf, int buf_size )
{
  int i;
  char action_buf[512];
  
  buf[0] = '\0';
  for ( i = 0; i < plan->num_ops; i++ ) {
    action_index_to_string( plan->ops[i], action_buf, sizeof(action_buf) );
    if ( i > 0 ) {
      strncat( buf, "\n", buf_size - strlen(buf) - 1 );
    }
    strncat( buf, action_buf, buf_size - strlen(buf) - 1 );
  }
}


/*
 * Copy a plan
 */
Plan copy_plan( Plan *src )
{
  Plan dest;
  int i;
  
  dest.num_ops = src->num_ops;
  dest.ops = (int *) calloc( dest.num_ops, sizeof(int) );
  for ( i = 0; i < dest.num_ops; i++ ) {
    dest.ops[i] = src->ops[i];
  }
  
  return dest;
}


/*
 * Free a plan
 */
void free_plan( Plan *plan )
{
  if ( plan && plan->ops ) {
    free( plan->ops );
    plan->ops = NULL;
    plan->num_ops = 0;
  }
}


/*
 * Find action index by string representation
 * Format: (action_name param1 param2 ...)
 * Returns -1 if not found
 */
int find_action_index_by_string( char *action_str )
{
  int i;
  char action_buf[512];
  
  if ( gnum_op_conn == 0 ) {
    /* Connectivity graph not built yet */
    return -1;
  }
  
  for ( i = 0; i < gnum_op_conn; i++ ) {
    action_index_to_string( i, action_buf, sizeof(action_buf) );
    if ( strcmp( action_str, action_buf ) == 0 ) {
      return i;
    }
  }
  
  return -1;
}


/*
 * Parse a single action line from disallowed plans file
 * Format: (action_name param1 param2 ...)
 * Returns action index, or -1 if not found
 */
int parse_action_line( char *line )
{
  int len;
  int result;
  
  /* Trim whitespace */
  while ( *line == ' ' || *line == '\t' ) line++;
  len = strlen( line );
  while ( len > 0 && ( line[len-1] == ' ' || line[len-1] == '\t' || 
                       line[len-1] == '\n' || line[len-1] == '\r' ) ) {
    line[--len] = '\0';
  }
  
  /* Skip empty lines */
  if ( len == 0 ) {
    return -2; /* Special value for empty line */
  }
  
  result = find_action_index_by_string( line );
  if ( result < 0 && gcmd_line.display_info >= 1 ) {
    printf( "Warning: Action not found in domain: %s\n", line );
  }
  
  return result;
}


/*
 * Load disallowed plans from file
 * Format: one action per line, plans separated by blank lines
 */
void load_disallowed_plans( void )
{
  FILE *fp;
  char line[512];
  char filepath[MAX_LENGTH * 2];
  int action_idx;
  int *temp_ops;
  int temp_num_ops = 0;
  int temp_max_ops = 100;
  
  if ( !gcmd_line.disallowed_plans_file[0] ) {
    if ( gcmd_line.display_info >= 1 ) {
      printf( "\nNo disallowed plans file specified - all plans allowed\n" );
    }
    return; /* No disallowed plans file specified */
  }
  
  sprintf( filepath, "%s%s", gcmd_line.path, gcmd_line.disallowed_plans_file );
  fp = fopen( filepath, "r" );
  if ( !fp ) {
    printf( "\nWarning: Could not open disallowed plans file: %s\n", filepath );
    return;
  }
  
  printf( "\nLoading disallowed plans from: %s\n", filepath );
  
  /* Initialize storage */
  gmax_disallowed_plans = 100;
  gdisallowed_plans = (Plan *) calloc( gmax_disallowed_plans, sizeof(Plan) );
  gnum_disallowed_plans = 0;
  
  temp_ops = (int *) calloc( temp_max_ops, sizeof(int) );
  
  while ( fgets( line, sizeof(line), fp ) != NULL ) {
    action_idx = parse_action_line( line );
    
    if ( action_idx == -2 ) {
      /* Empty line - end of current plan */
      if ( temp_num_ops > 0 ) {
        if ( gnum_disallowed_plans >= gmax_disallowed_plans ) {
          gmax_disallowed_plans *= 2;
          gdisallowed_plans = (Plan *) realloc( gdisallowed_plans, 
                                                gmax_disallowed_plans * sizeof(Plan) );
        }
        gdisallowed_plans[gnum_disallowed_plans].num_ops = temp_num_ops;
        gdisallowed_plans[gnum_disallowed_plans].ops = (int *) calloc( temp_num_ops, sizeof(int) );
        for ( action_idx = 0; action_idx < temp_num_ops; action_idx++ ) {
          gdisallowed_plans[gnum_disallowed_plans].ops[action_idx] = temp_ops[action_idx];
        }
        gnum_disallowed_plans++;
        temp_num_ops = 0;
      }
    } else if ( action_idx >= 0 ) {
      /* Valid action */
      if ( temp_num_ops >= temp_max_ops ) {
        temp_max_ops *= 2;
        temp_ops = (int *) realloc( temp_ops, temp_max_ops * sizeof(int) );
      }
      temp_ops[temp_num_ops++] = action_idx;
    }
    /* If action_idx == -1, action not found - skip it */
  }
  
  /* Handle last plan if file doesn't end with blank line */
  if ( temp_num_ops > 0 ) {
    if ( gnum_disallowed_plans >= gmax_disallowed_plans ) {
      gmax_disallowed_plans *= 2;
      gdisallowed_plans = (Plan *) realloc( gdisallowed_plans, 
                                            gmax_disallowed_plans * sizeof(Plan) );
    }
    gdisallowed_plans[gnum_disallowed_plans].num_ops = temp_num_ops;
    gdisallowed_plans[gnum_disallowed_plans].ops = (int *) calloc( temp_num_ops, sizeof(int) );
    for ( action_idx = 0; action_idx < temp_num_ops; action_idx++ ) {
      gdisallowed_plans[gnum_disallowed_plans].ops[action_idx] = temp_ops[action_idx];
    }
    gnum_disallowed_plans++;
  }
  
  free( temp_ops );
  fclose( fp );
  
  if ( gnum_disallowed_plans == 0 ) {
    printf( "\nNo disallowed plans found in file (or file is empty - all plans allowed)\n" );
  } else {
    printf( "\nLoaded %d disallowed plan(s)\n", gnum_disallowed_plans );
  }
}


/*
 * Compare two plans for equality
 * Returns TRUE if plans are identical, FALSE otherwise
 */
Bool plans_equal( Plan *p1, Plan *p2 )
{
  int i;
  
  if ( p1->num_ops != p2->num_ops ) {
    return FALSE;
  }
  
  for ( i = 0; i < p1->num_ops; i++ ) {
    if ( p1->ops[i] != p2->ops[i] ) {
      return FALSE;
    }
  }
  
  return TRUE;
}


/*
 * Check if a plan is disallowed
 * Returns TRUE if plan matches any disallowed plan, FALSE otherwise
 */
Bool is_plan_disallowed( Plan *plan )
{
  int i;
  char plan_buf[8192];
  char disallowed_buf[8192];
  
  if ( gnum_disallowed_plans == 0 ) {
    return FALSE; /* No disallowed plans - all plans allowed */
  }
  
  for ( i = 0; i < gnum_disallowed_plans; i++ ) {
    if ( plans_equal( plan, &gdisallowed_plans[i] ) ) {
      if ( gcmd_line.display_info >= 1 ) {
	plan_to_string( plan, plan_buf, sizeof(plan_buf) );
	printf( "\nPlan matches disallowed plan %d:\n%s\n", i + 1, plan_buf );
      }
      return TRUE;
    }
  }
  
  return FALSE;
}


/*
 * Check if a plan is a duplicate of any already found plan
 * Returns TRUE if duplicate, FALSE otherwise
 */
Bool is_plan_duplicate( Plan *plan )
{
  int i;
  
  for ( i = 0; i < gnum_found_plans; i++ ) {
    if ( plans_equal( plan, &gfound_plans[i] ) ) {
      return TRUE;
    }
  }
  
  return FALSE;
}


/*
 * Save current plan (from gplan_ops) to found plans array
 * Returns TRUE if plan was saved, FALSE if it was disallowed or duplicate
 */
Bool save_current_plan( void )
{
  Plan plan;
  int i;
  
  if ( gnum_plan_ops == 0 ) {
    return FALSE;
  }
  
  /* Create plan from current gplan_ops */
  plan.num_ops = gnum_plan_ops;
  plan.ops = (int *) calloc( plan.num_ops, sizeof(int) );
  for ( i = 0; i < plan.num_ops; i++ ) {
    plan.ops[i] = gplan_ops[i];
  }
  
  /* Check if disallowed */
  if ( is_plan_disallowed( &plan ) ) {
    if ( gcmd_line.display_info >= 1 ) {
      printf( "\nPlan is disallowed, skipping...\n" );
    }
    free_plan( &plan );
    return FALSE;
  }
  
  /* Check if duplicate */
  if ( is_plan_duplicate( &plan ) ) {
    if ( gcmd_line.display_info >= 1 ) {
      printf( "\nPlan is duplicate, skipping...\n" );
    }
    free_plan( &plan );
    return FALSE;
  }
  
  /* Save plan */
  if ( gnum_found_plans >= gmax_found_plans ) {
    if ( gmax_found_plans == 0 ) {
      gmax_found_plans = 10;
    } else {
      gmax_found_plans *= 2;
    }
    gfound_plans = (Plan *) realloc( gfound_plans, gmax_found_plans * sizeof(Plan) );
  }
  
  gfound_plans[gnum_found_plans] = plan;
  gnum_found_plans++;
  
  return TRUE;
}








/*
 *  ----------------------------- HEADERS FOR PARSING ----------------------------
 * ( fns defined in the scan-* files )
 */







void get_fct_file_name( char *filename );
void load_ops_file( char *filename );
void load_fct_file( char *filename );











/*
 *  ----------------------------- MAIN ROUTINE ----------------------------
 */




struct tms lstart, lend;







int main( int argc, char *argv[] )

{

  /* resulting name for ops file
   */
  char ops_file[MAX_LENGTH] = "";
  /* same for fct file 
   */
  char fct_file[MAX_LENGTH] = "";
  
  struct tms start, end;

  State current_start, current_end;
  int i, j;
  Bool found_plan;


  times ( &lstart );


  /* command line treatment
   */
  if ( argc == 1 || ( argc == 2 && *++argv[0] == '?' ) ) {
    ff_usage();
    exit( 1 );
  }
  if ( !process_command_line( argc, argv ) ) {
    ff_usage();
    exit( 1 );
  }


  /* make file names
   */

  /* one input name missing
   */
  if ( !gcmd_line.ops_file_name || 
       !gcmd_line.fct_file_name ) {
    fprintf(stdout, "\nff: two input files needed\n\n");
    ff_usage();      
    exit( 1 );
  }
  /* add path info, complete file names will be stored in
   * ops_file and fct_file 
   */
  sprintf(ops_file, "%s%s", gcmd_line.path, gcmd_line.ops_file_name);
  sprintf(fct_file, "%s%s", gcmd_line.path, gcmd_line.fct_file_name);


  /* parse the input files
   */

  /* start parse & instantiation timing
   */
  times( &start );
  /* domain file (ops)
   */
  if ( gcmd_line.display_info >= 1 ) {
    printf("\nff: parsing domain file");
  } 
  /* it is important for the pddl language to define the domain before 
   * reading the problem 
   */
  load_ops_file( ops_file );
  /* problem file (facts)
   */  
  if ( gcmd_line.display_info >= 1 ) {
    printf(" ... done.\nff: parsing problem file"); 
  }
  load_fct_file( fct_file );
  if ( gcmd_line.display_info >= 1 ) {
    printf(" ... done.\n\n");
  }

  /* This is needed to get all types.
   */
  build_orig_constant_list();

  /* last step of parsing: see if it's an ADL domain!
   */
  if ( !make_adl_domain() ) {
    printf("\nff: this is not an ADL problem!");
    printf("\n    can't be handled by this version.\n\n");
    exit( 1 );
  }


  /* now instantiate operators;
   */


  /**************************
   * first do PREPROCESSING * 
   **************************/


  /* start by collecting all strings and thereby encoding 
   * the domain in integers.
   */
  encode_domain_in_integers();

  /* inertia preprocessing, first step:
   *   - collect inertia information
   *   - split initial state into
   *        _ arrays for individual predicates
   *        - arrays for all static relations
   *        - array containing non - static relations
   */
  do_inertia_preprocessing_step_1();

  /* normalize all PL1 formulae in domain description:
   * (goal, preconds and effect conditions)
   *   - simplify formula
   *   - expand quantifiers
   *   - NOTs down
   */
  normalize_all_wffs();

  /* translate negative preconds: introduce symmetric new predicate
   * NOT-p(..) (e.g., not-in(?ob) in briefcaseworld)
   */
  translate_negative_preconds();

  /* split domain in easy (disjunction of conjunctive preconds)
   * and hard (non DNF preconds) part, to apply 
   * different instantiation algorithms
   */
  split_domain();


  /***********************************************
   * PREPROCESSING FINISHED                      *
   *                                             *
   * NOW MULTIPLY PARAMETERS IN EFFECTIVE MANNER *
   ***********************************************/

  build_easy_action_templates();
  build_hard_action_templates();

  times( &end );
  TIME( gtempl_time );

  times( &start );

  /* perform reachability analysis in terms of relaxed 
   * fixpoint
   */
  perform_reachability_analysis();

  times( &end );
  TIME( greach_time );

  times( &start );

  /* collect the relevant facts and build final domain
   * and problem representations.
   */
  collect_relevant_facts();

  times( &end );
  TIME( grelev_time );

  times( &start );

  /* now build globally accessable connectivity graph
   */
  build_connectivity_graph();

  times( &end );
  TIME( gconn_time );
  

  /***********************************************************
   * we are finally through with preprocessing and can worry *
   * bout finding a plan instead.                            *
   ***********************************************************/

  /* Load disallowed plans if specified */
  load_disallowed_plans();

  times( &start );

  /* another quick preprocess: approximate goal orderings and split
   * goal set into sequence of smaller sets, the goal agenda
   */
  compute_goal_agenda();

  /* make space in plan states info, and relax
   */
  for ( i = 0; i < MAX_PLAN_LENGTH + 1; i++ ) {
    make_state( &(gplan_states[i]), gnum_ft_conn );
    gplan_states[i].max_F = gnum_ft_conn;
  }
  make_state( &current_start, gnum_ft_conn );
  current_start.max_F = gnum_ft_conn;
  make_state( &current_end, gnum_ft_conn );
  current_end.max_F = gnum_ft_conn;
  initialize_relax();


  source_to_dest( &(gplan_states[0]), &ginitial_state );

  /* Initialize found plans storage */
  gmax_found_plans = gcmd_line.num_plans;
  if ( gmax_found_plans < 1 ) {
    gmax_found_plans = 1;
  }
  gfound_plans = (Plan *) calloc( gmax_found_plans, sizeof(Plan) );
  gnum_found_plans = 0;

  /* Search for K plans */
  /* Try enforced hill climbing first - if it fails, use best-first search */
  source_to_dest( &current_start, &ginitial_state );
  source_to_dest( &current_end, &(ggoal_agenda[0]) );

  found_plan = FALSE;
  
  /* Try enforced hill climbing */
  for ( i = 0; i < gnum_goal_agenda; i++ ) {
    if ( !do_enforced_hill_climbing( &current_start, &current_end ) ) {
      break;
    }
    source_to_dest( &current_start, &(gplan_states[gnum_plan_ops]) );
    if ( i < gnum_goal_agenda - 1 ) {
      for ( j = 0; j < ggoal_agenda[i+1].num_F; j++ ) {
        current_end.F[current_end.num_F++] = ggoal_agenda[i+1].F[j];
      }
    }
  }
  found_plan = ( i == gnum_goal_agenda ) ? TRUE : FALSE;

  if ( found_plan ) {
    /* Save the first plan if it's valid */
    if ( save_current_plan() ) {
      printf("\nFound and saved plan %d of %d requested\n", gnum_found_plans, gcmd_line.num_plans);
    } else {
      printf("\nFound plan was disallowed or duplicate.\n");
    }
    
    /* If we need more plans, use best-first search to find additional ones */
    if ( gnum_found_plans < gcmd_line.num_plans ) {
      printf("\nSearching for additional plans using Best-first Search...\n");
      do_best_first_search();
    }
  } else {
    /* EHC failed - use best-first search to find plans */
    printf("\n\nEnforced Hill-climbing failed !");
    printf("\nswitching to Best-first Search now.\n");
    do_best_first_search();
  }
  
  if ( gnum_found_plans < gcmd_line.num_plans ) {
    printf("\nFound %d plan(s) out of %d requested.\n", gnum_found_plans, gcmd_line.num_plans);
  }

  times( &end );
  TIME( gsearch_time );

  if ( gnum_found_plans > 0 ) {
    printf("\n\nFound %d plan(s):\n", gnum_found_plans);
    for ( i = 0; i < gnum_found_plans; i++ ) {
      printf("\n--- Plan %d ---\n", i + 1);
      /* Temporarily set gplan_ops to print this plan */
      gnum_plan_ops = gfound_plans[i].num_ops;
      for ( j = 0; j < gnum_plan_ops; j++ ) {
        gplan_ops[j] = gfound_plans[i].ops[j];
      }
      print_plan();
    }
  } else {
    printf("\n\nNo valid plans found.\n");
  }

  output_planner_info();

  printf("\n\n");
  exit( 0 );

}











/*
 *  ----------------------------- HELPING FUNCTIONS ----------------------------
 */












void output_planner_info( void )

{

  printf( "\n\ntime spent: %7.2f seconds instantiating %d easy, %d hard action templates", 
	  gtempl_time, gnum_easy_templates, gnum_hard_mixed_operators );
  printf( "\n            %7.2f seconds reachability analysis, yielding %d facts and %d actions", 
	  greach_time, gnum_pp_facts, gnum_actions );
  printf( "\n            %7.2f seconds creating final representation with %d relevant facts", 
	  grelev_time, gnum_relevant_facts );
  printf( "\n            %7.2f seconds building connectivity graph",
	  gconn_time );
  printf( "\n            %7.2f seconds searching, evaluating %d states, to a max depth of %d", 
	  gsearch_time, gevaluated_states, gmax_search_depth );
  printf( "\n            %7.2f seconds total time", 
	  gtempl_time + greach_time + grelev_time + gconn_time + gsearch_time );

  printf("\n\n");
  print_official_result();

  exit( 0 );


}


FILE *out;

void print_official_result( void )

{

  int i, k;
  char name[MAX_LENGTH];
  int saved_plan_ops[MAX_PLAN_LENGTH];
  int saved_num_plan_ops;

  sprintf( name, "%s.soln", gcmd_line.fct_file_name );

  if ( (out = fopen( name, "w")) == NULL ) {
    printf("\n\nCan't open official output file!\n\n");
    return;
  }

  times( &lend );
  fprintf(out, "Time %d\n", 
	 (int) ((lend.tms_utime - lstart.tms_utime + lend.tms_stime - lstart.tms_stime) * 10.0));

  /* Save current plan state */
  saved_num_plan_ops = gnum_plan_ops;
  for ( i = 0; i < gnum_plan_ops; i++ ) {
    saved_plan_ops[i] = gplan_ops[i];
  }

  /* Output all found plans */
  if ( gnum_found_plans > 0 ) {
    fprintf(out, "Found %d plan(s)\n\n", gnum_found_plans);
    for ( k = 0; k < gnum_found_plans; k++ ) {
      fprintf(out, "--- Plan %d ---\n", k + 1);
      
      /* Set gplan_ops to current plan for printing */
      gnum_plan_ops = gfound_plans[k].num_ops;
      for ( i = 0; i < gnum_plan_ops; i++ ) {
        gplan_ops[i] = gfound_plans[k].ops[i];
      }
      
      /*** Modified by Siddharth ***/
      printf("*** Plan %d States ***\n", k + 1);
      for ( i = 0; i < gnum_plan_ops; i++ ) {
        print_official_op_name( gplan_ops[i] );
        fprintf(out, "\n");
        printf("State \n");
        /* Note: gplan_states won't be correct for all plans, but we output what we have */
        if ( i < MAX_PLAN_LENGTH ) {
          print_state(gplan_states[i]);
          printf("\n\n");
        }
      }
      fprintf(out, "\n");
    }
  } else {
    /* Fallback to original plan if no plans were saved */
    gnum_plan_ops = saved_num_plan_ops;
    for ( i = 0; i < gnum_plan_ops; i++ ) {
      gplan_ops[i] = saved_plan_ops[i];
    }
    
    /*** Modified by Siddharth ***/
    printf("*** States ***\n");
    for ( i = 0; i < gnum_plan_ops+1; i++ ) {
      print_official_op_name( gplan_ops[i] );
      fprintf(out, "\n");
      printf("State \n");
      print_state(gplan_states[i]);
      printf("\n\n");
    }
  }

  /* Restore original plan state */
  gnum_plan_ops = saved_num_plan_ops;
  for ( i = 0; i < gnum_plan_ops; i++ ) {
    gplan_ops[i] = saved_plan_ops[i];
  }

  fclose( out );

}



void print_official_op_name( int index )

{

  int i;
  Action *a = gop_conn[index].action;

  if ( a->norm_operator ||
       a->pseudo_action ) {
    fprintf(out, "(%s", a->name ); 
    for ( i = 0; i < a->num_name_vars; i++ ) {
      fprintf(out, " %s", gconstants[a->name_inst_table[i]]);
    }
    fprintf(out, ")");
  }

}



void ff_usage( void )

{

  printf("\nusage of ff:\n");

  printf("\nOPTIONS   DESCRIPTIONS\n\n");
  printf("-p <str>    path for operator and fact file\n");
  printf("-o <str>    operator file name\n");
  printf("-f <str>    fact file name\n");
  printf("-k <num>    number of plans to find (default: 1)\n");
  printf("-a <file>   disallowed plans file (optional)\n\n");
  printf("-i <num>    run-time information level( preset: 1 )\n");
  printf("      0     only times\n");
  printf("      1     problem name, planning process infos\n");
  printf("    101     parsed problem data\n");
  printf("    102     cleaned up ADL problem\n");
  printf("    103     collected string tables\n");
  printf("    104     encoded domain\n");
  printf("    105     predicates inertia info\n");
  printf("    106     splitted initial state\n");
  printf("    107     domain with Wff s normalized\n");
  printf("    108     domain with NOT conds translated\n");
  printf("    109     splitted domain\n");
  printf("    110     cleaned up easy domain\n");
  printf("    111     unaries encoded easy domain\n");
  printf("    112     effects multiplied easy domain\n");
  printf("    113     inertia removed easy domain\n");
  printf("    114     easy action templates\n");
  printf("    115     cleaned up hard domain representation\n");
  printf("    116     mixed hard domain representation\n");
  printf("    117     final hard domain representation\n");
  printf("    118     reachability analysis results\n");
  printf("    119     facts selected as relevant\n");
  printf("    120     final domain and problem representations\n");
  printf("    121     connectivity graph\n");
  printf("    122     fixpoint result on each evaluated state\n");
  printf("    123     1P extracted on each evaluated state\n");
  printf("    124     H set collected for each evaluated state\n");
  printf("    125     False sets of goals <GAM>\n");
  printf("    126     detected ordering constraints leq_h <GAM>\n");
  printf("    127     the Goal Agenda <GAM>\n");



/*   printf("    109     reachability analysis results\n"); */
/*   printf("    110     final domain representation\n"); */
/*   printf("    111     connectivity graph\n"); */
/*   printf("    112     False sets of goals <GAM>\n"); */
/*   printf("    113     detected ordering constraints leq_h <GAM>\n"); */
/*   printf("    114     the Goal Agenda <GAM>\n"); */
/*   printf("    115     fixpoint result on each evaluated state <1Ph>\n"); */
/*   printf("    116     1P extracted on each evaluated state <1Ph>\n"); */
/*   printf("    117     H set collected for each evaluated state <1Ph>\n"); */
  
  printf("\n-d <num>    switch on debugging\n");
  printf("Also prints states.\n\n");

}



Bool process_command_line( int argc, char *argv[] )

{

  char option;

  gcmd_line.display_info = 1;
  gcmd_line.debug = 0;
  gcmd_line.num_plans = 1;
  
  memset(gcmd_line.ops_file_name, 0, MAX_LENGTH);
  memset(gcmd_line.fct_file_name, 0, MAX_LENGTH);
  memset(gcmd_line.path, 0, MAX_LENGTH);
  memset(gcmd_line.disallowed_plans_file, 0, MAX_LENGTH);

  while ( --argc && ++argv ) {
    if ( *argv[0] != '-' || strlen(*argv) != 2 ) {
      return FALSE;
    }
    option = *++argv[0];
    switch ( option ) {
    default:
      if ( --argc && ++argv ) {
	switch ( option ) {
	case 'p':
	  strncpy( gcmd_line.path, *argv, MAX_LENGTH );
	  break;
	case 'o':
	  strncpy( gcmd_line.ops_file_name, *argv, MAX_LENGTH );
	  break;
	case 'f':
	  strncpy( gcmd_line.fct_file_name, *argv, MAX_LENGTH );
	  break;
	case 'i':
	  sscanf( *argv, "%d", &gcmd_line.display_info );
	  break;
	case 'k':
	  sscanf( *argv, "%d", &gcmd_line.num_plans );
	  if ( gcmd_line.num_plans < 1 ) {
	    printf( "\nff: number of plans must be at least 1\n\n" );
	    return FALSE;
	  }
	  break;
	case 'a':
	case 'd':
	  /* Support both -a and -d for disallowed plans file */
	  strncpy( gcmd_line.disallowed_plans_file, *argv, MAX_LENGTH );
	  gcmd_line.disallowed_plans_file[MAX_LENGTH - 1] = '\0'; /* Ensure null termination */
	  break;
	default:
	  printf( "\nff: unknown option: %c entered\n\n", option );
	  return FALSE;
	}
      } else {
	return FALSE;
      }
    }
  }

  return TRUE;

}

