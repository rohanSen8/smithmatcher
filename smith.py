import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import arange
import cmath
class SmithChart:


  def __init__(self, name, age):
      self.name = name
      self.age = age
      self.figure, self.axes = plt.subplots()
      self.plotBackgroundplots(self.axes)
      '''
      z=self.gammatoZ(0)
      print(z)
      gamma=complex(0,1)
      plt.plot(gamma.real,gamma.imag,'bo')
      z = self.gammatoZ(gamma)
      print(z)
      '''
      #plt.show(block=False)
      #plt.hold(True)
  def showMoves(self):
      plt.show()
  def addxl(self,xl):
      z=self.gammatoZ(self.gamma)
      center=complex(z.real/(1+z.real),0)
      radius=1/(1+z.real)
      v1=self.gamma-center
      z=z+complex(0,xl)
      r1,phi1=cmath.polar(v1)
      #v1.real*v2.real+v1.imag*v2.imag


      self.gamma=self.Ztogamma(z)
      v2 = self.gamma - center
      r2, phi2 = cmath.polar(v2)

      if xl>0:
          pac = mpatches.Arc([center.real,center.imag],2*radius, 2 * radius, angle=0, theta1=(phi2)*180/cmath.pi, theta2=phi1*180/cmath.pi,color='r')
      else:
          pac = mpatches.Arc([center.real, center.imag], 2 * radius, 2 * radius, angle=0,
                             theta1=(phi1) * 180 / cmath.pi, theta2=phi2 * 180 / cmath.pi, color='r')



      self.axes.add_patch(pac)

      self.axes.plot(self.gamma.real, self.gamma.imag, 'bo')
      #self.axes.plot(center.real, center.imag, 'bo')
      #self.axes.plot(v1.real, v1.imag, 'ro')
      #self.axes.plot(v2.real, v2.imag, 'ro')
      #plt.show()

  def addrin(self, rin):
      z = self.gammatoZ(self.gamma)
      center = complex(1, 1/z.imag)
      radius = 1 / (z.imag)
      v1 = self.gamma - center
      r1, phi1 = cmath.polar(v1)
      z = z + complex(rin, 0)
      self.gamma = self.Ztogamma(z)
      v2 = self.gamma - center
      r2, phi2 = cmath.polar(v2)
      self.axes.plot(self.gamma.real, self.gamma.imag, 'bo')

      if z.imag>0:
          pac = mpatches.Arc([center.real,center.imag],2*radius, 2 * radius, angle=0, theta1=(phi1)*180/cmath.pi, theta2=phi2*180/cmath.pi,color='r')
      else:
          pac = mpatches.Arc([center.real, center.imag], 2 * radius, 2 * radius, angle=0,
                             theta1=(phi2+cmath.pi) * 180 / cmath.pi, theta2=(phi1+cmath.pi) * 180 / cmath.pi, color='r')
      self.axes.add_patch(pac)
      self.axes.plot(center.real, center.imag, 'bo')
      self.axes.plot(v1.real, v1.imag, 'ro')
      self.axes.plot(v2.real, v2.imag, 'ro')
      print(phi1,phi2)
      plt.show()
  def hitzyOne(self):
      '''
      err=10
      while err>0.01:
          self.z
    '''
      z=self.gammatoZ(self.gamma)
      k=z.real/(1+z.real)
      temp=(0.25-k*k/(z.real*z.real))/(k+0.5)
      gammar=0.5*(temp+k-0.5)
      print(gammar)
      gammai=-np.sqrt(0.25-(gammar+0.5)*(gammar+0.5))
      #self.gamma=complex(gammar,gammai)

      #self.axes.plot(gammar,gammai,'go')
      return self.gammatoZ(complex(gammar,gammai))-z
  def hitzyMatch(self):
      '''
      err=10
      while err>0.01:
          self.z
    '''
      z=self.gammatoZ(self.gamma)

      return complex(1,0)-z
  def addZ(self,rin, xl):
      z = self.gammatoZ(self.gamma)

      z = z + complex(rin, xl)
      self.gamma = self.Ztogamma(z)
      self.axes.plot(self.gamma.real, self.gamma.imag, 'bo')
      # plt.show()
  def moveTowardGenerator(self,linLambda):
      ph=cmath.rect(1, -2 * 2 * cmath.pi * linLambda)
      print(ph)
      r, angGamma = cmath.polar(self.gamma)
      self.gamma=self.gamma*ph
      r1,anGammaNew=cmath.polar(self.gamma)
      print(r,angGamma,r1,anGammaNew)
      if linLambda>0:
          pac = mpatches.Arc([0, 0], 2 * r, 2 * r, angle=0, theta1=anGammaNew * 180 / cmath.pi,
                             theta2=angGamma * 180 / cmath.pi, color='r')
      else:
          pac = mpatches.Arc([0, 0], 2 * r, 2 * r, angle=0, theta1=angGamma * 180 / cmath.pi,
                             theta2=anGammaNew * 180 / cmath.pi, color='r')

      self.axes.add_patch(pac)
      self.axes.plot(self.gamma.real, self.gamma.imag, 'bo')
      #plt.show()
  def intializeZin(self,zin):
      self.gamma=self.Ztogamma(zin)
      self.axes.plot(self.gamma.real, self.gamma.imag, 'bo')
      #plt.plot()
      #plt.show()
  def ZA2AZ(self):
      self.gamma=-self.gamma
      x_values = [self.gamma.real, -self.gamma.real]
      y_values = [self.gamma.imag,-self.gamma.imag]
      plt.plot(x_values, y_values, 'r', linestyle="--")
      self.axes.plot(self.gamma.real, self.gamma.imag, 'bo')
      # plt.plot()
      #plt.show()
  def gammatoZ(self,gamma):
      zl=(gamma+1)/(1-gamma)
      return zl
  def Ztogamma(self,zl):
      gamma=(zl-1)/(1+zl)
      return gamma
  def plotBackgroundplots(self, axes):
      plt.title('Colored Circle')
      #plt.axis([-1, 1, -1, 1])
      plt.axis([-3, 3, -3, 3])

      self.AddCircle(axes, 0, 0, 1)
      rl = 1
      self.AddCircle(axes, rl / (1 + rl), 0, 1 / (rl + 1))
      self.AddCircle(axes, -rl / (1 + rl), 0, 1 / (rl + 1),yellow=True)
      xl = arange(-3, 3, 0.5)
      for x in xl:
          self.drawxl(axes, x)

  def plotArc(self,axes, radius, center, ang, xl):
      if xl > 0:
          pac = mpatches.Arc(list(center), 2 * radius, 2 * radius, angle=270 - ang, theta1=0, theta2=ang)
      else:
          pac = mpatches.Arc(list(center), 2 * radius, 2 * radius, angle=90, theta1=0, theta2=ang)
      axes.add_artist(pac)

  def AddCircle(self,axes, centerx, centery, radius,yellow=False):
      if yellow:
          Drawing_colored_circle = plt.Circle((centerx, centery), radius, fill=False,color='y')
      else:
        Drawing_colored_circle = plt.Circle((centerx, centery), radius, fill=False)

      axes.add_artist(Drawing_colored_circle)

  def genIntersection(self,xl):
      s = tuple([((xl * xl - 1) / (xl * xl + 1)), (2 * xl / (xl * xl + 1))])
      return s

  def genAngle(self,r, xl):
      if xl > 0:
          s = tuple([r[0] - 1, r[1] - 1 / xl])
          sc = (-s[1]) / (np.sqrt(s[0] * s[0] + s[1] * s[1]))
      else:
          s = tuple([r[0] - 1, r[1] - 1 / xl])
          sc = (s[1]) / (np.sqrt(s[0] * s[0] + s[1] * s[1]))

      ang = np.arccos(sc)
      return ang

  def drawxl(self,axes, xl):
      if xl != 0:
          r = self.genIntersection(xl)
          ang = np.degrees(self.genAngle(r, xl))
          center = tuple([1, 1 / xl])
          radius = abs(1 / xl)
          #print(ang)
          #plt.plot(r[0], r[1], 'bo')
          self.plotArc(axes, radius, center, ang, xl)
      else:
          plt.plot([1, -1], [0, 0], color='black')