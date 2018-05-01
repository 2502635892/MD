
void Scale_Velocity(AtomInfo *atoms, int N)
{

  int i = 0;
  double ene = 0.0;
  double scalar = 1.0;

#pragma omp parallel for reduction(+ : ene)
  for (i = 0; i < N; i++)
  {
    ene = ene + atoms[i].mass * (atoms[i].velocity[0] * atoms[i].velocity[0] + atoms[i].velocity[1] * atoms[i].velocity[1] + atoms[i].velocity[2] * atoms[i].velocity[2]);
  }

  scalar = sqrt(Termp * Dim * K_B * N / ene); // scale coefficient

#pragma omp parallel for
  for (i = 0; i < N; i++)
  {
    atoms[i].velocity[0] = atoms[i].velocity[0] * scalar;
    atoms[i].velocity[1] = atoms[i].velocity[1] * scalar;
    atoms[i].velocity[2] = atoms[i].velocity[2] * scalar;
  }

} //scale the velocity when it not equilibrium




void initialize(AtomInfo *atoms, int N, int nxyz[], double a[], double T)
{
  double r0[FCC_NUM][Dim] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int m = 0;
  int i = 0;
  int id = 0;

  for (nx = 0; nx < nxyz[0]; nx++)
  {
    for (ny = 0; ny < nxyz[1]; ny++)
    {
      for (nz = 0; nz < nxyz[2]; nz++)
      {
        for (m = 0; m < FCC_NUM; m++)
        {
          atoms[id].atom_id = id;
          atoms[id].position[0] = a[0] * (nx + r0[m][0]);
          atoms[id].position[1] = a[1] * (ny + r0[m][1]);
          atoms[id].position[2] = a[2] * (nz + r0[m][2]);
          id = id + 1;
        }
      }
    }

  } //initialize the position of atoms, how to parallel here ?

#pragma omp parallel for
  for (i = 0; i < N; ++i)
  {
    atoms[i].mass = Mass;
  } //initialize the mass of atoms

#pragma omp parallel for
  for (i = 0; i < N; ++i)
  {
    atoms[i].velocity[0] = (rand() / (RAND_MAX + 1.0)) - 0.5;
    atoms[i].velocity[1] = (rand() / (RAND_MAX + 1.0)) - 0.5;
    atoms[i].velocity[2] = (rand() / (RAND_MAX + 1.0)) - 0.5;
  }

  double momentum_average[Dim] = {0.0, 0.0, 0.0};
  double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;

#pragma omp parallel for reduction(+ : sum0, sum1, sum2)
  for (i = 0; i < N; ++i)
  {
    sum0 = sum0 + atoms[i].velocity[0] * atoms[i].mass;
    sum1 = sum1 + atoms[i].velocity[1] * atoms[i].mass;
    sum2 = sum2 + atoms[i].velocity[2] * atoms[i].mass;
  }

  momentum_average[0] = sum0 / N;
  momentum_average[1] = sum1 / N;
  momentum_average[2] = sum2 / N;

#pragma omp parallel for
  for (i = 0; i < N; ++i)
  {
    atoms[i].velocity[0] = atoms[i].velocity[0] - momentum_average[0] / atoms[i].mass;
    atoms[i].velocity[1] = atoms[i].velocity[1] - momentum_average[1] / atoms[i].mass;
    atoms[i].velocity[2] = atoms[i].velocity[2] - momentum_average[2] / atoms[i].mass;
  }

  Scale_Velocity(atoms, N); // scale the velocity

} //initialize information of atoms

